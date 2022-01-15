from utils import CharErrorRate, WordErrorRate
import torch
import torch.nn as nn
import torchaudio.functional as aF
import torch.nn.functional as F
from torch.distributions import Categorical
from fairseq.models.wav2vec import Wav2VecModel
from tqdm import tqdm
from copy import deepcopy
import numpy as np
# from pesq import pesq
from utils import Wav2Letter, MelCepstralDistortion, WordErrorRate, CharErrorRate, preprocess
from dataset import SPEAKERS


try:
    import apex.amp as amp
    AMP = True
except ImportError:
    AMP = False
    
def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell

class VQ_Wav2Vec(nn.Module):
    def __init__(self, checkpoint_path):
        super(VQ_Wav2Vec, self).__init__()
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.encoder = Wav2VecModel.build_model(checkpoint['args'], task=None)
        self.encoder.load_state_dict(checkpoint['model'])

    def forward(self, audio):
        self.encoder.train()
        return self.encoder(audio)

    def encode(self, audio):
        self.encoder.eval()
        z = self.encoder.feature_extractor(audio)
        _, idxs = self.encoder.vector_quantizer.forward_idx(z)
        del z
        return idxs
    
class RnnDecoder(nn.Module):
    def __init__(self, code_book_num, code_embedding_dim, n_speakers, speaker_embedding_dim, rnn_layers_num,
                 conditioning_channels, mu_embedding_dim, rnn_channels,
                 fc_channels, bits, hop_length):
        super(RnnDecoder, self).__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.hop_length = hop_length
        self.rnn_layers_num = rnn_layers_num
        self.code_embedding_1 = nn.Embedding(code_book_num, code_embedding_dim)
        self.code_embedding_2 = nn.Embedding(code_book_num, code_embedding_dim)
        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
        self.rnn1 = nn.GRU(2*code_embedding_dim + speaker_embedding_dim, conditioning_channels, num_layers=2, batch_first=True, bidirectional=True)
        self.rnn1A = nn.ModuleList([ nn.GRU(2*code_embedding_dim + 2*conditioning_channels, conditioning_channels, num_layers=2, batch_first=True, bidirectional=True) for _ in range(rnn_layers_num) ])
        self.mu_embedding = nn.Embedding(self.quantization_channels, mu_embedding_dim)
        self.rnn2 = nn.GRU(mu_embedding_dim + 2*conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)

    def forward(self, x, idxs1, idxs2, speakers, audio_size):
        z1 = self.code_embedding_1(idxs1)
        z2 = self.code_embedding_2(idxs2)
        z = torch.cat((z1, z2), dim=2)
        speakers = self.speaker_embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)
        z = torch.cat((z, speakers), dim=-1)

        # -- Conditioning sub-network
        z, _ = self.rnn1(z)
        ## GRU Block
        for i in range(self.rnn_layers_num):
            z = torch.cat((z, z1, z2), dim=2)
            z, _ = self.rnn1A[i](z)

        z = F.interpolate(z.transpose(1, 2), size=audio_size)
        z = z.transpose(1, 2)

        # -- Autoregressive model
        x = self.mu_embedding(x)
        x, _ = self.rnn2(torch.cat((x, z), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, idxs1, idxs2, speaker, hop_length):
        cell = get_gru_cell(self.rnn2)
        z1 = self.code_embedding_1(idxs1)
        z2 = self.code_embedding_2(idxs2)
        z = torch.cat((z1, z2), dim=2)

        speaker = self.speaker_embedding(speaker)
        speaker = speaker.unsqueeze(1).expand(-1, z.size(1), -1)
        z = torch.cat((z, speaker), dim=-1)
        z, _ = self.rnn1(z)

        for i in range(self.rnn_layers_num):
            z = torch.cat((z, z1, z2), dim=2)
            z, _ = self.rnn1A[i](z)

        z = F.interpolate(z.transpose(1, 2), scale_factor=hop_length)
        z = z.transpose(1, 2)

        batch_size, sample_size, _ = z.size()
        h = torch.zeros(batch_size, self.rnn_channels, device=z.device)
        x = torch.zeros(batch_size, device=z.device).fill_(self.quantization_channels // 2).long()
        unbind = torch.unbind(z, dim=1)
        outputs = torch.empty(len(unbind))
        for i, m in enumerate(tqdm(unbind, leave=False)):
            x = self.mu_embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)
            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            dist = Categorical(logits=logits)
            x = dist.sample()
            outputs[i] = x.squeeze().float()
        outputs = aF.mu_law_decoding(outputs.float(), self.quantization_channels)
        return outputs

class VQW2V_RNNDecoder(nn.Module):
    def __init__(self, enc_checkpoint_path, decoder_cfg, device):
        super(VQW2V_RNNDecoder, self).__init__()
        self.encoder = VQ_Wav2Vec(enc_checkpoint_path)
        self.decoder = RnnDecoder(**decoder_cfg)
        self.quantization_channels = 2**decoder_cfg['bits']
        self.hop_length = decoder_cfg['hop_length']
        self.device = device
        self.encoder.eval()

    def forward(self, audio, mu_audio, speakers):
        with torch.no_grad():
            idxs = self.encoder.encode(audio[:, :-1])
            idxs1, idxs2 = idxs[:,:,0], idxs[:,:,1]
        output = self.decoder(mu_audio[:, :-1], idxs1, idxs2, speakers, mu_audio.size(1)-1)
        return output
    
    def training_step(self, batch, batch_idx):
        self.encoder.eval()
        self.decoder.train()
        audio, speakers = batch['audio'].to(self.device), batch['speakers'].to(self.device)
        mu_audio =  aF.mu_law_encoding(audio, self.quantization_channels).long()
        output = self(audio, mu_audio, speakers)
        loss = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:])
        return { 'loss' : loss }

    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        self.decoder.eval()
        audio, speakers = batch['audio'].to(self.device), batch['speakers'].to(self.device)
        mu_audio =  aF.mu_law_encoding(audio, self.quantization_channels).long()
        with torch.no_grad():
            logits = self(audio, mu_audio, speakers)
            loss = F.cross_entropy(logits.transpose(1, 2), mu_audio[:, 1:])
            dist = Categorical(logits=logits)
            outputs = dist.sample().squeeze()
            cv = aF.mu_law_decoding(outputs.float(), self.quantization_channels)
        return { 'val_loss' : loss.unsqueeze(0), 'cv' : cv.cpu().detach(), 'tar' : audio.cpu().detach() }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.cat([ out['val_loss'] for out in outputs ], dim=0)).item()
        target_audio = [ out['tar'].cpu().numpy() for out in outputs ]
        cv_audio = [ out['cv'].cpu().numpy() for out in outputs ]
        mcd = MelCepstralDistortion()
        for i in range(len(target_audio)):
            l = len(target_audio[i])
            for j in range(l):
                tar  = target_audio[i][j]
                cv = cv_audio[i][j]
                mcd.calculate_metric(cv, tar)
        avg_mcd = mcd.compute()
        logs = { 'avg_val_loss' : avg_loss, 'avg_mcd' : avg_mcd }
        return { 'avg_mcd' : avg_mcd, 'log' : logs }
        # return { 'avg_loss' : avg_loss, 'log' : logs }

    def convert(self, audio, speakers):
        with torch.no_grad():
            idxs = self.encoder.encode(audio)
            idxs1, idxs2 = idxs[:, :, 0], idxs[:, :, 1]
            output = self.decoder.generate(idxs1, idxs2, speakers, self.hop_length) 
        return output, idxs

    def conversion_step(self, batch, batch_idx, rep=False):
        # check batch size is 1.
        self.encoder.eval()
        self.decoder.eval()
        audio, speakers = batch['source_audio'].to(self.device), batch['target_speaker_id'].to(self.device)
        assert len(audio) == 1, "Batch size in the conversion step must be 1."
        output, idxs = self.convert(audio, speakers)
        ret = dict(
            cv=output.cpu().detach(),
            converted_audio_path=batch['converted_audio_path'][0],
            target_speaker_id=batch['target_speaker_id'][0],
            target_audio=batch['target_audio'][0],
            utterance=batch['utterance'][0]
        )
        if rep:
            ret['representation'] = idxs[0].cpu().detach()
        return ret

    def conversion_epoch_end(self, outputs):
        converted_audio_paths = np.array([ out['converted_audio_path'] for out in outputs ]).flatten()
        target_speakers = [ SPEAKERS[out['target_speaker_id']] for out in outputs ]
        target_speaker_set = list(set(target_speakers))
        target_audio = [ out['target_audio'].cpu().numpy() for out in outputs ]
        utterances = np.array([ out['utterance'] for out in outputs ]).flatten()
        w2l = Wav2Letter(self.device)
        mcd = MelCepstralDistortion(target_speaker_set)
        wer = WordErrorRate(target_speaker_set)
        cer = CharErrorRate(target_speaker_set)

        all_logs = []

        eval_num = len(converted_audio_paths)
        for i in tqdm(range(eval_num)):
            converted_audio_path = converted_audio_paths[i]
            target_speaker = target_speakers[i]
            tar = target_audio[i]
            utterance = preprocess(utterances[i])
            cv = outputs[i]['cv']

            transcription = preprocess(w2l.decode(cv))
            
            mcd_value = mcd.calculate_metric(cv, tar, target_speaker)
            wer_value = wer.calculate_metric(transcription, utterance, target_speaker)
            cer_value = cer.calculate_metric(transcription, utterance, target_speaker)

            all_logs.append({ 
                'converted_audio_path' : converted_audio_path,
                'reference' : utterance,
                'transcription' : transcription,
                'mcd' : mcd_value,
                'wer' : wer_value,
                'cer' : cer_value
            })

            del target_speaker
            del tar
            del utterance
            del cv
            del transcription

        avg_mcd = mcd.compute()
        avg_wer = wer.compute()
        avg_cer = cer.compute()

        logs = { 'avg_mcd' : avg_mcd, 'avg_wer' : avg_wer, 'avg_cer' : avg_cer }
        return { 'logs' : logs, 'all_logs' : all_logs }
    
    def state_dict(self, optimizer, scheduler):
        dic =  {
            "decoder": deepcopy(self.decoder.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()),
            "scheduler": deepcopy(scheduler.state_dict()),
        }
        if AMP:
            dic['amp'] = deepcopy(amp.state_dict())
        return dic

    def load_model(self, checkpoint, amp=True):
        self.decoder.load_state_dict(checkpoint["decoder"])
        if amp:
            amp.load_state_dict(checkpoint["amp"])
    

class VQW2V_RNNDecoder_Replay(VQW2V_RNNDecoder):
    def __init__(self, enc_checkpoint_path, decoder_cfg, device):
        super(VQW2V_RNNDecoder_Replay, self).__init__(enc_checkpoint_path, decoder_cfg, device)

    def forward(self, audio, mu_audio, speakers):
        with torch.no_grad():
            idxs = self.encoder.encode(audio[:, :-1])
            idxs1, idxs2 = idxs[:,:,0], idxs[:,:,1]
        output = self.decoder(mu_audio[:, :-1], idxs1, idxs2, speakers, mu_audio.size(1)-1)
        return output
    
    def training_step(self, batch_fine, batch_pre, batch_idx):
        self.encoder.eval()
        self.decoder.train()
        ### fine
        fine_audio, fine_speakers = batch_fine['audio'].to(self.device), batch_fine['speakers'].to(self.device)
        # adjust tensor size
        #fine_audio = fine_audio.view(-1, fine_audio.size(-1))
        #fine_speakers = fine_speakers.flatten() 
        fine_mu_audio =  aF.mu_law_encoding(fine_audio, self.quantization_channels)

        # encode
        with torch.no_grad():
            fine_idxs = self.encoder.encode(fine_audio[:, :-1])
            fine_idxs1, fine_idxs2 = fine_idxs[:,:,0], fine_idxs[:,:,1]
        # decode
        fine_output = self.decoder(fine_mu_audio[:, :-1], fine_idxs1, fine_idxs2, fine_speakers,fine_mu_audio.size(1)-1)

        ###
        ### pre
        pre_audio, pre_speakers = batch_pre['audio'].to(self.device), batch_pre['speakers'].to(self.device)
        #pre_audio = pre_audio.view(-1, pre_audio.size(-1))
        pre_mu_audio =  aF.mu_law_encoding(pre_audio, self.quantization_channels)
        #pre_speakers = pre_speakers.flatten()

        if 'idxs' in batch_pre:
            pre_idxs = batch_pre['idxs'].to(self.device)
            # pre_idxs = pre_idxs.view(-1, pre_idxs.size(-2), pre_idxs.size(-1))
            pre_idxs1, pre_idxs2 = pre_idxs[:,:,0], pre_idxs[:,:,1]
        else:
            with torch.no_grad():
                pre_idxs = self.encoder.encode(pre_audio[:, :-1])
                pre_idxs1, pre_idxs2 = pre_idxs[:,:,0], pre_idxs[:,:,1]
        # decode
        pre_output = self.decoder(pre_mu_audio[:, :-1], pre_idxs1, pre_idxs2, pre_speakers, pre_mu_audio.size(1)-1)

        # calculate loss
        output = torch.cat([ fine_output, pre_output], dim=0)
        mu_audio = torch.cat( [ fine_mu_audio, pre_mu_audio], dim=0)
        loss = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:])

        return { 'loss' : loss }

    def validation_step(self, batch_fine, batch_pre, batch_idx):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            ### fine
            fine_audio, fine_speakers = batch_fine['audio'].to(self.device), batch_fine['speakers'].to(self.device)
            # adjust tensor size
            #fine_audio = fine_audio.view(-1, fine_audio.size(-1))
            #fine_speakers = fine_speakers.flatten() 
            fine_mu_audio =  aF.mu_law_encoding(fine_audio, self.quantization_channels)

            # encode
            
            fine_idxs = self.encoder.encode(fine_audio[:, :-1])
            fine_idxs1, fine_idxs2 = fine_idxs[:,:,0], fine_idxs[:,:,1]
            
            # decode
            fine_output = self.decoder(fine_mu_audio[:, :-1], fine_idxs1, fine_idxs2, fine_speakers,fine_mu_audio.size(1)-1)

            ###
            ### pre
            pre_audio, pre_speakers = batch_pre['audio'].to(self.device), batch_pre['speakers'].to(self.device)
            #pre_audio = pre_audio.view(-1, pre_audio.size(-1))
            pre_mu_audio =  aF.mu_law_encoding(pre_audio, self.quantization_channels)
            #pre_speakers = pre_speakers.flatten()

            if 'idxs' in batch_pre:
                pre_idxs = batch_pre['idxs'].to(self.device)
                #pre_idxs = pre_idxs.view(-1, pre_idxs.size(-2), pre_idxs.size(-1))
                pre_idxs1, pre_idxs2 = pre_idxs[:,:,0], pre_idxs[:,:,1]
            else:
                
                pre_idxs = self.encoder.encode(pre_audio[:, :-1])
                pre_idxs1, pre_idxs2 = pre_idxs[:,:,0], pre_idxs[:,:,1]
            # decode
            pre_output = self.decoder(pre_mu_audio[:, :-1], pre_idxs1, pre_idxs2, pre_speakers, pre_mu_audio.size(1)-1)
            ## calculate loss
            output = torch.cat([ fine_output, pre_output], dim=0)
            mu_audio = torch.cat( [ fine_mu_audio, pre_mu_audio], dim=0)
            loss = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:])
            # --

        return { 'val_loss' : loss.unsqueeze(0) }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.cat([ out['val_loss'] for out in outputs ], dim=0)).item()
        logs = { 'avg_val_loss' : avg_loss }
        return { 'avg_loss' : avg_loss, 'log' : logs }
