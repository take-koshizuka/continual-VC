import os
import torch
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from torchaudio.models import wav2vec2_large_lv60k
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np
import re
import pysptk
import pyworld as pw
import Levenshtein as Lev
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

PATH_TO_ASR = "checkpoints/wav2vec2-base-960h.pt"

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
    def __init__(self, sr=16000, order=24):
        self.sr = sr
        self.order = order
        self.mcd = 0
        self.n_frames = 0

    def wav2mcep(self, wav):
        wav = wav / np.abs(wav).max() * 0.999
        f0, time = pw.harvest(wav.astype(np.float64), self.sr)
        sp = pw.cheaptrick(wav.astype(np.float64), f0, time, self.sr)
        alpha = pysptk.util.mcepalpha(self.sr)
        mc = pysptk.sp2mc(sp, order=self.order, alpha=alpha)[:, 1:]
        return mc
    
    def calculate_metric(self, wav, wav_ref):
        mcd_inst, n_frames = self.mcd_calc(wav, wav_ref)
        self.mcd += mcd_inst
        self.n_frames += n_frames

    def compute(self):
        mcd = float(self.mcd) / self.n_frames
        return mcd
    
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


class CharErrorRate:
    def __init__(self):
        self.cer = 0
        self.n_chars = 0

    def calculate_metric(self, transcript, reference):
        cer_inst = self.cer_calc(transcript, reference)
        self.cer += cer_inst
        self.n_chars += len(reference.replace(' ', ''))

    def compute(self):
        cer = float(self.cer) / self.n_chars
        return cer * 100
    
    def cer_calc(self, s1, s2):
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

class WordErrorRate:
    def __init__(self):
        self.wer = 0
        self.n_words = 0

    def calculate_metric(self, transcript, reference):
        wer_inst = self.wer_calc(transcript, reference)
        self.wer += wer_inst
        self.n_words += len(reference.split())

    def compute(self):
        wer = float(self.wer) / self.n_words
        return wer * 100
    
    def wer_calc(self, s1, s2):
        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))