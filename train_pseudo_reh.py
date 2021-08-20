from model import VQW2V_RNNDecoder_PseudoRehearsal
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import WavDataset, PseudoWavDataset
from model import VQW2V_RNNDecoder_PseudoRehearsal
from utils import EarlyStopping
from copy import deepcopy
from tqdm import tqdm

try:
    import apex.amp as amp
    AMP = True
except ImportError:
    AMP = False

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(train_config_path, resume_path, checkpoint_path, scalars_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(train_config_path, 'r') as f:
        cfg = json.load(f)

    fix_seed(cfg['seed'])

    tr_ds_fine = WavDataset(
        root=cfg['dataset']['folder_in_archive'],
        data_list_path=cfg['dataset']['train_list_path'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        bits=cfg['dataset']['bits'],
    )

    va_ds_fine = WavDataset(
        root=cfg['dataset']['folder_in_archive'],
        data_list_path=cfg['dataset']['val_list_path'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        bits=cfg['dataset']['bits']
    )

    tr_dl_fine = DataLoader(tr_ds_fine,
            batch_size=cfg['dataset']['batch_size_fine'],
            shuffle=True,
            num_workers=cfg['dataset']['n_workers'],
            drop_last=True)

    va_dl_fine = DataLoader(va_ds_fine,
            batch_size=cfg['dataset']['batch_size_fine'],
            num_workers=cfg['dataset']['n_workers'],
            drop_last=False)

    tr_ds_pre = PseudoWavDataset(
        root=cfg['dataset']['folder_in_archive'],
        data_list_path=cfg['dataset']['train_list_path'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        bits=cfg['dataset']['bits'],
    )

    va_ds_pre = PseudoWavDataset(
        root=cfg['dataset']['folder_in_archive'],
        data_list_path=cfg['dataset']['val_list_path'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        bits=cfg['dataset']['bits']
    )

    tr_dl_pre = DataLoader(tr_ds_pre,
            batch_size=cfg['dataset']['batch_size_pre'],
            shuffle=True,
            num_workers=cfg['dataset']['n_workers'],
            drop_last=True)
    tr_it_pre = iter(tr_dl_pre)

    va_dl_pre = DataLoader(va_ds_pre,
            batch_size=cfg['dataset']['batch_size_pre'],
            num_workers=cfg['dataset']['n_workers'],
            drop_last=False)
    va_it_pre = iter(va_dl_pre)

    model = VQW2V_RNNDecoder_PseudoRehearsal(cfg['encoder'], cfg['decoder'], device)
    model.to(device)

    optimizer = optim.Adam(
                    model.decoder.parameters(),
                    lr=cfg['optim']['lr'],
                    betas=(float(cfg['optim']['beta_0']), float(cfg['optim']['beta_1']))
                )

    schedular = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=cfg['scheduler']['milestones'],
                    gamma=cfg['scheduler']['gamma']
                )

    early_stopping = EarlyStopping('avg_loss', 'min')

    init_epochs = 1
    max_epochs = cfg['epochs']

    if not resume_path is None:
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_model(checkpoint)
        model.to(device)
        init_epochs = checkpoint['epochs']

    if AMP:
        model.decoder, optimizer = amp.initialize(model.decoder, optimizer, opt_level="01")

    model.encoder.eval()
    writer = SummaryWriter()
    
    for i in tqdm(range(init_epochs, max_epochs + 1)):
        # training phase
        for batch_idx, train_batch_fine in enumerate(tqdm(tr_dl_fine, leave=False)):
            try:
                train_batch_pre = next(tr_it_pre) 
            except StopIteration:
                tr_it_pre = iter(tr_dl_pre)
                train_batch_pre = next(tr_it_pre)

            out = model.training_step(train_batch_fine, train_batch_pre, batch_idx)
            if AMP:
                with amp.scale_loss(out['loss'], optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            else:
                out['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1)

            optimizer.step()
            schedular.step()
            optimizer.zero_grad()
            del train_batch_fine
            del train_batch_pre

        # validation phase
        outputs = []
        for batch_idx, eval_batch_fine in enumerate(zip(va_dl_fine, leave=False)):
            try:
                eval_batch_pre = next(va_it_pre) 
            except StopIteration:
                va_it_pre = iter(va_dl_pre)
                eval_batch_pre = next(va_it_pre)

            out = model.validation_step(eval_batch_fine, eval_batch_pre, batch_idx)
            outputs.append(out)
            del eval_batch_fine
            del eval_batch_pre

        val_result = model.validation_epoch_end(outputs)

        writer.add_scalars('data/loss', val_result['log'])

        # early_stopping
        if early_stopping.judge(val_result):
            early_stopping.update(val_result)
            state_dict = model.state_dict()
            state_dict['epochs'] = i
            early_stopping.best_state = deepcopy(state_dict)
    
    best_state = early_stopping.best_state
    torch.save(best_state, checkpoint_path)
    writer.export_scalars_to_json(scalars_path)
    writer.close()

if __name__ == '__main__':
    train_config_path = "config/train_pseudo_reh_config.json"
    resume_path = None # or path
    checkpoint_path = ""
    scalars_path = ""
    main(train_config_path, resume_path, checkpoint_path, scalars_path)