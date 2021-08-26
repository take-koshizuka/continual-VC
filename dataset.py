import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import re
import csv
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
    def __init__(self, root, data_list_path, sr, sample_frames, hop_length, bits):
        self.root = Path(root)
        self.sr = sr
        self.sample_frames = sample_frames
        self.hop_length = hop_length
        self.bits = bits
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
        audio = audio / np.abs(audio).max() * 0.999
        pos = random.randint(0, (len(audio) // self.hop_length) - self.sample_frames - 2)
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]
        return dict(audio=torch.FloatTensor(audio), speakers=metadata['speaker_id'])

class PseudoWavDataset(Dataset):
    def __init__(self, root, pseudo_data_list_path, sr, sample_frames, hop_length, bits):
        self.root = Path(root)
        self.sr = sr
        self.sample_frames = sample_frames
        self.hop_length = hop_length
        self.bits = bits

        with open(pseudo_data_list_path) as file:
            metadata = json.load(file)
            self.metadata = [
                {
                    'speaker_id' : SPEAKERS.index(speaker_id),
                    'audio_path' : str(self.root / Path("wav") / Path(filename).with_suffix(".wav")),
                    'representation_path' : str(self.root / Path("rep") / Path(filename).with_suffix(".npy")),
                }
                for speaker_id, filename in metadata
            ]
            self.labels = [ id for id, _ in metadata ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        audio, _ = librosa.load(metadata['audio_path'], sr=self.sr)
        idxs  = np.load(metadata['representation_path'])
        hop_length  = len(audio) // len(idxs) 
        pos = random.randint(0, (len(audio) // hop_length) - self.sample_frames - 2)
        audio = audio[pos * hop_length:(pos + self.sample_frames) * hop_length + 1]
        idxs = idxs[pos : pos + self.sample_frames ,:]
        return dict(audio=torch.FloatTensor(audio),
                    idxs=idxs,
                    peakers=metadata['speaker_id'])

class ConversionDataset(Dataset):
    def __init__(self, root, outdir, synthesis_list_path, sr):
        self.root = Path(root)
        self.outdir = Path(outdir)
        self.sr = sr
        with open(synthesis_list_path) as f:
            metadata = json.load(f)
        
        ut = Utterance(self.root)
        self.metadata = []
        for source_speaker, target_speaker, filename, converted_audio_path in metadata:
            meta_item = {
                'source_speaker_id' : SPEAKERS.index(source_speaker), 
                'source_audio_path' : str(self.root / Path(f"cmu_us_{source_speaker}_arctic") / Path("wav") / Path(filename).with_suffix(".wav")),
                'target_speaker_id' : SPEAKERS.index(target_speaker), 
                'target_audio_path' : str(self.root / Path(f"cmu_us_{target_speaker}_arctic") / Path("wav") / Path(filename).with_suffix(".wav")), 
                'converted_audio_path' : str(self.outdir / converted_audio_path),
                'utterance' : ut.filename2utterance[filename]
            }
            self.metadata.append(meta_item)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        meta_item = self.metadata[index]
        source_audio, _ = librosa.load(meta_item['source_audio_path'], sr=self.sr)
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

def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)