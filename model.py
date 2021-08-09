from json import decoder
import torch
import torch.nn as nn
import torchaudio.functional as aF
import torch.nn.functional as F
from torch.distributions import Categorical
from fairseq.models.wav2vec import Wav2VecModel
from tqdm import tqdm
from copy import deepcopy

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

    def generate(self, idxs1, idxs2, speaker, audio_size):
        output = []
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

        z = F.interpolate(z.transpose(1, 2), size=audio_size)
        z = z.transpose(1, 2)

        batch_size, sample_size, _ = z.size()
        h = torch.zeros(batch_size, self.rnn_channels, device=z.device)
        x = torch.zeros(batch_size, device=z.device).fill_(self.quantization_channels // 2).long()
        unbind = torch.unbind(z, dim=1)

        for m in tqdm(unbind, leave=False):
            x = self.mu_embedding(x)
            h = cell(torch.cat((x, m), dim=1), h)
            x = F.relu(self.fc1(h))
            logits = self.fc2(x)
            dist = Categorical(logits=logits)
            x = dist.sample()
            output.append(2 * x.float().item() / (self.quantization_channels - 1.) - 1.)

        output = aF.mu_law_decoding(output, self.quantization_channels)
        return output

class VQW2V_RNNDecoder(nn.Module):
    def __init__(self, enc_checkpoint_path, decoder_cfg, device):
        super(VQW2V_RNNDecoder, self).__init__()
        self.encoder = VQ_Wav2Vec(enc_checkpoint_path)
        self.decoder = RnnDecoder(**decoder_cfg)
        self.quantization_channels = 2**decoder_cfg['bits']
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
        mu_audio =  aF.mu_law_encoding(audio, self.quantization_channels)
        output = self(audio, mu_audio, speakers)
        loss = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:])
        return { 'loss' : loss }

    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        self.decoder.eval()
        audio, speakers = batch['audio'].to(self.device), batch['speakers'].to(self.device)
        mu_audio =  aF.mu_law_encoding(audio, self.quantization_channels)
        with torch.no_grad():
            output = self(audio, mu_audio, speakers)
            loss = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:])
        return { 'val_loss' : loss.unsqueeze(0) }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.cat([ out['val_loss'] for out in outputs ], dim=0)).item()
        tensorboard_logs = { 'avg_val_loss' : avg_loss }
        return { 'avg_loss' : avg_loss, 'log' : tensorboard_logs }

    def convert(self, batch, batch_idx):
        self.encoder.eval()
        self.decoder.eval()
        audio, speakers = batch['audio'].to(self.device), batch['speakers'].to(self.device)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        assert audio.dim() == 3
        with torch.no_grad():
            idxs = self.encoder.encode(audio)
            idxs1, idxs2 = idxs[:, :, 0], idxs[:, :, 1]
            output = self.decoder.generate(idxs1, idxs2, speakers, audio.size(0))
            
        return { 'output' : output }
    
    def state_dict(self):
        dic =  {
            "decoder": deepcopy(self.decoder.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "schedular": deepcopy(self.schedular.state_dict()),
        }
        if AMP:
            dic['amp'] = deepcopy(amp.state_dict())
        return dic

    def load_model(self, checkpoint):
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.schedular.load_state_dict(checkpoint["schedular"])
        if AMP:
            amp.load_state_dict(checkpoint["amp"])


class VQW2V_RNNDecoder_PseudoRehearsal(VQW2V_RNNDecoder):
    def __init__(self, enc_checkpoint_path, decoder_cfg, device):
        super(VQW2V_RNNDecoder_PseudoRehearsal, self).__init__(enc_checkpoint_path, decoder_cfg, device)

    def forward(self, audio, mu_audio, speakers):
        with torch.no_grad():
            idxs = self.encoder.encode(audio[:, :-1])
            idxs1, idxs2 = idxs[:,:,0], idxs[:,:,1]
        output = self.decoder(mu_audio[:, :-1], idxs1, idxs2, speakers, mu_audio.size(1)-1)
        return output
    
    def training_step(self, batch_fine, batch_pre, batch_idx):
        self.encoder.eval()
        self.decoder.train()
        audio, speakers = batch_fine['audio'].to(self.device), batch_fine['speakers'].to(self.device)
        mu_audio =  aF.mu_law_encoding(audio, self.quantization_channels)
        output = self(audio, mu_audio, speakers)
        loss_fine = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:], reduce='none')

        pre_mu_audio, pre_speakers = batch_pre['pre_mu_audio'].to(self.device), batch_pre['pre_speakers'].to(self.device)
        pre_idxs1, pre_idxs2, = batch_pre['pre_idxs1'].to(self.device), batch_pre['pre_idxs2'].to(self.device)
        pre_output = self.decoder(pre_mu_audio[:, :-1], pre_idxs1, pre_idxs2, pre_speakers, pre_mu_audio.size(1)-1)
        loss_pre = F.cross_entropy(pre_output.transpose(1, 2), pre_mu_audio[:, 1:], reduce='none')
        loss = torch.mean(torch.cat([loss_fine, loss_pre]))

        return { 'loss' : loss, 'loss_fine' : loss_fine.mean(), 'loss_past' : loss_pre.mean() }

    def validation_step(self,  batch_fine, batch_pre, batch_idx):
        self.encoder.eval()
        self.decoder.eval()
        audio, speakers = batch_fine['audio'].to(self.device), batch_fine['speakers'].to(self.device)
        mu_audio =  aF.mu_law_encoding(audio, self.quantization_channels)
        output = self(audio, mu_audio, speakers)
        loss_fine = F.cross_entropy(output.transpose(1, 2), mu_audio[:, 1:], reduce='none')

        pre_mu_audio, pre_speakers = batch_pre['pre_mu_audio'].to(self.device), batch_pre['pre_speakers'].to(self.device)
        pre_idxs1, pre_idxs2, = batch_pre['pre_idxs1'].to(self.device), batch_pre['pre_idxs2'].to(self.device)
        pre_output = self.decoder(pre_mu_audio[:, :-1], pre_idxs1, pre_idxs2, pre_speakers, pre_mu_audio.size(1)-1)
        loss_pre = F.cross_entropy(pre_output.transpose(1, 2), pre_mu_audio[:, 1:], reduce='none')
        loss = torch.mean(torch.cat([loss_fine, loss_pre]))
        return { 'val_loss' : loss.unsqueeze(0), 'val_loss_fine' : loss_fine.mean(), 'val_loss_past' : loss_pre.mean() }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.cat([ out['val_loss'] for out in outputs ], dim=0)).item()
        avg_loss_fine = torch.mean(torch.cat([ out['val_loss_fine'] for out in outputs ], dim=0)).item()
        avg_loss_past = torch.mean(torch.cat([ out['val_loss_past'] for out in outputs ], dim=0)).item()
        tensorboard_logs = { 'avg_val_loss' : avg_loss, 'avg_loss_fine' : avg_loss_fine, 'avg_loss_past' : avg_loss_past }
        return { 'avg_loss' : avg_loss, 'log' : tensorboard_logs }
    
