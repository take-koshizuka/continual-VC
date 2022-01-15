import os
import json
import torch
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from torchaudio.models import wav2vec2_large_lv60k
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np
import pysptk
import jiwer
import pyworld as pw
import Levenshtein as Lev
import collections
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pathlib import Path



PATH_TO_ASR = "checkpoints/wav2vec2-base-960h.pt"

def get_metadata(path_list_dir, speakers, train_size, val_size, random_split=True):
    # path_list_dir: data_list or pseudo_data_list
    train_metadata = []
    val_metadata = []
    
    path_list = [  Path(path_list_dir) / f"train_val_{sp}.json" for sp in speakers ]
    for path in path_list:
        with open(str(path)) as f:
            lst = json.load(f)

        if random_split:
            np.random.shuffle(lst)
        
        train_metadata.extend(lst[:train_size])
        val_metadata.extend(lst[train_size:(train_size+val_size)])

    return train_metadata, val_metadata

def get_metadata_test(path_list_dir, speakers):
    # path_list_dir: data_list or pseudo_data_list
    test_metadata = []
    path_list = [  Path(path_list_dir) / f"test_{sp}.json" for sp in speakers ]
    for path in path_list:
        with open(str(path)) as f:
            lst = json.load(f)
        
        test_metadata.extend(lst)
    return test_metadata

def get_metadata_generate(root, source, targets, size):
    metadata = []
    path = (Path(root) / Path(f"cmu_us_{source}_arctic/wav"))
    wav_list = path.rglob('*.wav')
    wav_list = [ file_path.stem for file_path in wav_list ]

    for target in targets:
        np.random.shuffle(wav_list)
        clipped_wav_list = wav_list[:size]
        for wav_name in clipped_wav_list:
            metadata.append([
                source,
                target,
                wav_name,
                f"{target}-{wav_name}"
            ])
    return metadata

class EarlyStopping(object):
    def __init__(self, monitor='loss', direction='min'):
        self.monitor = monitor
        self.direction = direction
        self.best_state = None
        if direction == 'min':
            self.monitor_values = { self.monitor : float('inf') }
        elif direction == 'max':
            self.monitor_values = { self.monitor : -float('inf') }
        else:
            raise ValueError("args: [direction] must be min or max")

    def judge(self, values):
        return (self.direction == 'min' and self.monitor_values[self.monitor] > values[self.monitor]) \
                    or (self.direction == 'max' and self.monitor_values[self.monitor] < values[self.monitor])

    def update(self, values):
        self.monitor_values[self.monitor] = values[self.monitor]


class Wav2Letter:
    def __init__(self, device):
        if not os.path.exists(PATH_TO_ASR):
            original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
            model = import_huggingface_model(original)
            torch.save(model.state_dict(), PATH_TO_ASR)
        
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = wav2vec2_large_lv60k(num_out=32)
        self.model.load_state_dict(torch.load(PATH_TO_ASR))
        self.model.to(device)
        self.device = device
        
    def decode(self, audio):
        input_values = self.tokenizer(audio, return_tensors = "pt").input_values
        input_values = input_values.to(self.device)
        logits = self.model(input_values)[0]
        prediction = torch.argmax(logits, dim = -1).cpu()
        transcription = self.tokenizer.batch_decode(prediction)[0]
        return transcription

class MelCepstralDistortion:
    def __init__(self, target_speakers=None, sr=16000, order=24):
        self.sr = sr
        self.order = order
        self.mcd = 0
        self.n_frames = 0
        self.target_speakers = target_speakers
        if not target_speakers is None:
            self.score_per_speaker = { sp : dict(mcd=0, n_frames=0) for sp in target_speakers }
        

    def wav2mcep(self, wav):
        wav = wav / np.abs(wav).max() * 0.999
        _f0, time = pw.harvest(wav.astype(np.float64), self.sr)
        f0 = pw.stonemask(wav.astype(np.float64), _f0, time, self.sr)
        sp = pw.cheaptrick(wav.astype(np.float64), f0, time, self.sr)
        alpha = pysptk.util.mcepalpha(self.sr)
        mc = pysptk.sp2mc(sp, order=self.order, alpha=alpha)[:, 1:]
        return mc
    
    def calculate_metric(self, wav, wav_ref, target_speaker=None):
        mcd_inst, n_frames = self.mcd_calc(wav, wav_ref)
        self.mcd += mcd_inst
        self.n_frames += n_frames
        if not target_speaker is None:
            self.score_per_speaker[target_speaker]['mcd'] += mcd_inst
            self.score_per_speaker[target_speaker]['n_frames'] += n_frames

        return mcd_inst / n_frames

    def compute(self):
        if not self.target_speakers is None:
            scores = { sp: float(self.score_per_speaker[sp]['mcd']) / self.score_per_speaker[sp]['n_frames'] for sp in self.target_speakers }
        scores['all'] = float(self.mcd) / self.n_frames
        return scores
    
    def mcd_calc(self, wav, wav_ref):
        mc = self.wav2mcep(wav)
        mc_ref = self.wav2mcep(wav_ref)
        _, path = fastdtw(mc, mc_ref, dist=euclidean)
        twf = np.array(path).T
        
        mc_dtw = mc[twf[0]]
        mc_ref_dtw = mc_ref[twf[1]]

        diff2sum = np.sum((mc_dtw - mc_ref_dtw)**2, 1)
        
        mcd_value = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        n_frames = len(diff2sum)
        return mcd_value * n_frames, n_frames

def preprocess(text):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.SentencesToListOfWords(word_delimiter=" "),
        jiwer.RemoveEmptyStrings(),
    ])
    words_list = transformation(text)
    return " ".join(words_list)

class CharErrorRate:
    def __init__(self, target_speakers=None):
        self.cer = 0
        self.n_chars = 0
        self.target_speakers = target_speakers
        if not target_speakers is None:
            self.score_per_speaker = { sp : dict(cer=0, n_chars=0) for sp in target_speakers }

    def calculate_metric(self, transcript, reference, target_speaker=None):
        cer_inst = self.cer_calc(transcript, reference)
        self.cer += cer_inst
        self.n_chars += len(reference.replace(' ', ''))
        if not target_speaker is None:
            self.score_per_speaker[target_speaker]['cer'] += cer_inst
            self.score_per_speaker[target_speaker]['n_chars'] += len(reference.replace(' ', ''))
        return cer_inst / len(reference.replace(' ', ''))

    def compute(self):
        if not self.target_speakers is None:
            scores = { sp: (float(self.score_per_speaker[sp]['cer']) / self.score_per_speaker[sp]['n_chars'])*100 for sp in self.target_speakers }
        scores['all'] = (float(self.cer) / self.n_chars)*100
        return scores
    
    def cer_calc(self, s1, s2):
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

class WordErrorRate:
    def __init__(self, target_speakers=None):
        self.wer = 0
        self.n_words = 0
        self.target_speakers = target_speakers
        if not target_speakers is None:
            self.score_per_speaker = { sp : dict(wer=0, n_words=0) for sp in target_speakers }

    def calculate_metric(self, transcript, reference, target_speaker=None):
        wer_inst = self.wer_calc(transcript, reference)
        self.wer += wer_inst
        self.n_words += len(reference.split())
        if not target_speaker is None:
            self.score_per_speaker[target_speaker]['wer'] += wer_inst
            self.score_per_speaker[target_speaker]['n_words'] += len(reference.split())
        return wer_inst / len(reference.split())

    def compute(self):
        if not self.target_speakers is None:
            scores = { sp: (float(self.score_per_speaker[sp]['wer']) / self.score_per_speaker[sp]['n_words'])*100 for sp in self.target_speakers }
        scores['all'] = (float(self.wer) / self.n_words) * 100
        return scores
    
    def wer_calc(self, s1, s2):
        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

