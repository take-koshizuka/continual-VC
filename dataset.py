from os import curdir
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import random
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data.sampler import BatchSampler

SPEAKERS = [
    "aew",
    "ahw",
    "aup",
    "awb",
    "axb",
    "bdl",
    "clb",
    "eey",
    "fem",
    "gka",
    "jmk",
    "ksp",
    "ljm",
    "lnh",
    "rms",
    "rxr",
    "slp",
    "slt"
]

class WavDataset(Dataset):
    def __init__(self, root, sr, sample_frames, hop_length, data_list_path=None, metadata=None):
        self.root = Path(root)
        self.sr = sr
        self.sample_frames = sample_frames
        self.hop_length = hop_length
        if not metadata is None:
            self.metadata = metadata
        else:
            with open(data_list_path) as file:
                metadata = json.load(file)
        
        self.metadata = [
            {
                'speaker_id' : SPEAKERS.index(speaker_id),
                'audio_path' : str(self.root / Path(f"cmu_us_{speaker_id}_arctic") / Path("wav") / Path(filename).with_suffix(".wav"))
            }
            for speaker_id, filename in metadata
        ]
        self.labels = [ id for id, _ in metadata ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        audio, _ = librosa.load(metadata['audio_path'], sr=self.sr)
        audio = audio / np.abs(audio).max() * (1.0 - 1e-8)
        pos = random.randint(0, (len(audio) // self.hop_length) - self.sample_frames - 2)
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]
        return dict(audio=torch.FloatTensor(audio), speakers=metadata['speaker_id'])

class PseudoWavDataset(Dataset):
    def __init__(self, root, sr, sample_frames, hop_length, data_list_path=None, metadata=None):
        self.root = Path(root)
        self.sr = sr
        self.sample_frames = sample_frames
        self.hop_length = hop_length
        self.metadata_cur = { }
        self.metadata_past = { }

        if not metadata is None:
            self.metadata = metadata
        else:
            with open(data_list_path) as file:
                metadata = json.load(file)

        self.speakers = list(set([ sp for sp, _ in metadata ]))
        self.metadata = [
            {
                'speaker_id' : SPEAKERS.index(speaker_id),
                'audio_path' : str(self.root / Path("wav") / Path(filename).with_suffix(".npy")),
                'representation_path' : str(self.root / Path("rep") / Path(filename).with_suffix(".npy")),
            }
            for speaker_id, filename in metadata
        ]
        self.labels = [ id for id, _ in metadata ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        audio = np.load(metadata['audio_path'])
        audio = audio / np.abs(audio).max() * (1.0 - 1e-8)
        idxs  = np.load(metadata['representation_path'])

        pos = random.randint(0, (len(audio) // self.hop_length) - self.sample_frames - 2)
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]
        idxs = idxs[pos : pos + self.sample_frames ,:]

        return dict(audio=audio, idxs=idxs, speakers=metadata['speaker_id'])


class ConversionDataset(Dataset):
    def __init__(self, root, outdir, sr, unlabeled=False, synthesis_list_path=None, metadata=None):
        self.root = Path(root)
        self.outdir = Path(outdir)
        self.sr = sr
        self.unlabeled = unlabeled

        if metadata is None:
            with open(synthesis_list_path) as file:
                metadata = json.load(file)

        self.load_utterance()
        self.metadata = []
        for source_speaker, target_speaker, filename, converted_audio_path in metadata:
            meta_item = {
                'source_speaker_id' : SPEAKERS.index(source_speaker),
                'source_audio_path' : str(self.root / Path(f"cmu_us_{source_speaker}_arctic") / Path("wav") / Path(filename).with_suffix(".wav")),
                'target_speaker_id' : SPEAKERS.index(target_speaker), 
                'target_audio_path' : str(self.root / Path(f"cmu_us_{target_speaker}_arctic") / Path("wav") / Path(filename).with_suffix(".wav")), 
                'converted_audio_path' : str(self.outdir / converted_audio_path),
                'utterance' : self.ut.get_utterance(filename)
            }
            self.metadata.append(meta_item)
    
    def load_utterance(self):
        self.ut = Utterance(self.root)
    
    def select_subset(self, ind):
        self.metadata = [ self.metadata[i] for i in ind ]
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        meta_item = self.metadata[index]
        source_audio, _ = librosa.load(meta_item['source_audio_path'], sr=self.sr)
        if self.unlabeled:
            target_audio = np.empty(1)
        else:
            target_audio, _ = librosa.load(meta_item['target_audio_path'], sr=self.sr)
        return dict(
            source_speaker_id=meta_item['source_speaker_id'],
            source_audio=torch.FloatTensor(source_audio),
            target_speaker_id=meta_item['target_speaker_id'],
            target_audio=torch.FloatTensor(target_audio),
            converted_audio_path=meta_item['converted_audio_path'],
            utterance=meta_item['utterance']
        )

class Utterance:
    def __init__(self, root):
        self.root = Path(root)
        self.filename2utterance = {}
        for speaker in SPEAKERS:
            text_path = self.root / f"cmu_us_{speaker}_arctic" / "etc" / "txt.done.data"
            if not text_path.exists():
                continue
            with open(text_path) as f:
                lines = f.readlines()
                for line in lines:
                    utterance = line.split('"')[1]
                    filename = line.split(" ")[1]
                    self.filename2utterance[filename] = utterance

    def get_utterance(self, filename):
        return self.filename2utterance[filename]


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_samples):
        self.labels = np.array(dataset.labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = len(self.labels_set)
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size