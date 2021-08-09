import torch
from torch.utils.data import Dataset
import re
import json
import random
import numpy as np
import librosa
from pathlib import Path
import torchaudio.functional as aF

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
                    'audio_path' : str(self.root / Path(f"cmu_us_{speaker_id}_arctic") / Path("wav") / Path(file_name).with_suffix(".wav"))
                }
                for speaker_id, file_name in metadata
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
                    'mu_audio_path' : mu_audio_path,
                    'idxs_path' : idxs_path
                }
                for speaker_id, mu_audio_path, idxs_path in metadata
            ]
            self.labels = [ id for id, _ in metadata ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        pre_mu_audio = np.load(metadata['mu_audio_path'])
        pre_idxs  = np.load(metadata['idxs_path'])
        pre_idxs1, pre_idxs2 = pre_idxs[:,:,0], pre_idxs[:,:,1]

        # idxとaudioのアラインメント


        return dict(past_mu_audio=torch.LongTensor(pre_mu_audio), 
                    past_idxs1= pre_idxs1, past_idxs2=pre_idxs2, 
                    past_speakers=metadata['speaker_id'])

class ConversionDataset(Dataset):
    def __init__(self, root, out_dir, speaker_list, synthesis_list, sr):
        self.root = Path(root)
        self.out_dir = Path(out_dir)
        self.sr = sr
        with open(self.root / speaker_list) as file:
            self.speakers = sorted(json.load(file))

        with open(self.root / synthesis_list) as file:
            metadata = json.load(file)
            corpus_pathset = { source_speaker_id: self.root / Path(f"cmu_us_{source_speaker_id}_arctic") / "etc/txt.done.data" for source_speaker_id, _, _, _, _ in metadata }
            self.corpus_set = { source_speaker_id: CorpusDatasetCMU(corpus_path) for source_speaker_id, corpus_path in corpus_pathset.items() }

            self.metadata = [
                {
                    'source_speaker_id' : self.speakers.index(source_speaker_id),
                    'source_path': str(self.root / Path(source_dir) / Path("wav") / Path(file_name).with_suffix(".wav")),
                    'target_speaker_id': self.speakers.index(target_speaker_id),
                    'target_path' : str(self.root / Path(target_dir) / Path("wav") / Path(file_name).with_suffix(".wav")),
                    'converted_path' : str(self.out_dir / Path(source_speaker_id + "2" + target_speaker_id) / Path(file_name).with_suffix(".wav")),
                    'word_list' : self.corpus_set[source_speaker_id].get_wordlist(file_name),
                    'char_list' : self.corpus_set[source_speaker_id].get_charlist(file_name)
                } for source_speaker_id, source_dir, target_speaker_id, target_dir, file_name in metadata
            ]

        converted_dir_set = set([ str(self.out_dir / Path(source_speaker_id + "2" + target_speaker_id)) \
                                    for source_speaker_id, source_dir, target_speaker_id, target_dir, file_name in metadata ])
        for dir in converted_dir_set:
            Path(dir).mkdir(exist_ok=True, parents=True)
        self.labels = [ (source_speaker_id, target_speaker_id) for source_speaker_id, _, target_speaker_id, _, _ in metadata ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        metadata = self.metadata[index]
        source_audio, _ = librosa.load(metadata['source_path'], sr=self.sr)
        target_audio, _ = librosa.load(metadata['target_path'], sr=self.sr)
        return metadata['source_path'], metadata['target_path'], torch.LongTensor([metadata['source_speaker_id']]), torch.FloatTensor(source_audio), \
                torch.LongTensor([metadata['target_speaker_id']]), torch.FloatTensor(target_audio), \
                metadata['converted_path'], metadata['word_list'], metadata['char_list']



def fix_abbreviation(s):
    s = s.replace('.', '').replace(',', '')
    s = re.sub('[-]+', ' ', s)
    s = s.replace('etc', 'et cetera')
    return s

class CorpusDatasetCMU:
    def __init__(self, corpus_path):
        self.filename2word = {}
        self.filename2char = {}
        with open(corpus_path) as f:
            lines = f.readlines()
            for line in lines:
                word = line.split('"')[1]
                word = fix_abbreviation(word)
                word_list = list(map(lambda x: x.upper(), word.split()))
                character_list = [ w.upper() for w in word.replace(' ', '') ]
                filename = line.split(" ")[1]
                self.filename2word[filename] = word_list
                self.filename2char[filename] = character_list

    def get_wordlist(self, filename):
        return self.filename2word[filename]

    def get_charlist(self, filename):
        return self.filename2char[filename]

