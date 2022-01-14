from os import curdir
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import random
import numpy as np
import librosa
from pathlib import Path

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


class BalanceWavDataset(Dataset):
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

        self.speakers = list(set([ sp for sp, _ in metadata ]))
        self.metadata = {}
        for sp in self.speakers:
            self.metadata[sp] = [
                {
                'speaker_id' : SPEAKERS.index(speaker_id),
                'audio_path' : str(self.root / Path(f"cmu_us_{speaker_id}_arctic") / Path("wav") / Path(filename).with_suffix(".wav"))
                }
                for speaker_id, filename in metadata if speaker_id == sp
            ]

    def __len__(self):
        L = min([ len(self.metadata[sp]) for sp in self.speakers ])
        return L

    def __getitem__(self, index):
        audio_bucket = []
        speaker_bucket = []
        for sp in self.speakers:
            metadata = self.metadata[sp][index]

            audio, _ = librosa.load(metadata['audio_path'], sr=self.sr)
            audio = audio / np.abs(audio).max() * (1.0 - 1e-8)

            pos = random.randint(0, (len(audio) // self.hop_length) - self.sample_frames - 2)
            audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]
            
            audio_bucket.append(torch.FloatTensor(audio).unsqueeze(0))
            speaker_bucket.append(torch.tensor([ metadata['speaker_id']] ).unsqueeze(0))

        audio_bucket = torch.cat(audio_bucket, dim=0)
        speaker_bucket = torch.cat(speaker_bucket, dim=0)
        return dict(audio=audio_bucket, speakers=speaker_bucket)


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
        self.metadata = {}
        
        for sp in self.speakers:
            self.metadata[sp] = [
                {
                    'speaker_id' : SPEAKERS.index(speaker_id),
                    'audio_path' : str(self.root / Path("wav") / Path(filename).with_suffix(".npy")),
                    'representation_path' : str(self.root / Path("rep") / Path(filename).with_suffix(".npy")),
                }
                for speaker_id, filename in metadata if speaker_id == sp
            ]

    def __len__(self):
        L = min([ len(self.metadata[sp]) for sp in self.speakers ])
        return L

    def __getitem__(self, index):
        audio_bucket = []
        speaker_bucket = []
        idxs_bucket = []
        for sp in self.speakers:
            metadata = self.metadata[sp][index]

            audio = np.load(metadata['audio_path'])
            audio = audio / np.abs(audio).max() * (1.0 - 1e-8)
            idxs  = np.load(metadata['representation_path'])

            pos = random.randint(0, (len(audio) // self.hop_length) - self.sample_frames - 2)
            audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]
            idxs = idxs[pos : pos + self.sample_frames ,:]

            audio_bucket.append(torch.FloatTensor(audio).unsqueeze(0))
            idxs_bucket.append(torch.LongTensor(idxs).unsqueeze(0))
            speaker_bucket.append(torch.tensor([ metadata['speaker_id']] ).unsqueeze(0))

        audio_bucket = torch.cat(audio_bucket, dim=0)
        idxs_bucket= torch.cat(idxs_bucket, dim=0)
        speaker_bucket = torch.cat(speaker_bucket, dim=0)

        return dict(audio=audio_bucket, idxs=idxs_bucket, speakers=speaker_bucket)


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
