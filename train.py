import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import WavDataset
from model import VQW2V_RNNDecoder
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

    tr_ds = WavDataset(
        root=cfg['dataset']['folder_in_archive'],
        data_list_path=cfg['dataset']['train_list_path'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        bits=cfg['dataset']['bits'],
    )

    va_ds = WavDataset(
        root=cfg['dataset']['folder_in_archive'],
        data_list_path=cfg['dataset']['val_list_path'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        bits=cfg['dataset']['bits']
    )

    tr_dl = DataLoader(tr_ds,
            batch_size=cfg['dataset']['batch_size'],
            shuffle=True,
            num_workers=cfg['dataset']['n_workers'],
            drop_last=True)

    va_dl = DataLoader(va_ds,
            batch_size=cfg['dataset']['batch_size'],
            num_workers=cfg['dataset']['n_workers'],
            drop_last=False)

    model = VQW2V_RNNDecoder(cfg['encoder'], cfg['decoder'], device)
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
        for batch_idx, train_batch in enumerate(tr_dl):
            out = model.training_step(train_batch, batch_idx)
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
            del train_batch

        # validation phase
        outputs = []
        for batch_idx, eval_batch in enumerate(va_dl):
            out = model.validation_step(eval_batch, batch_idx)
            outputs.append(out)
            del eval_batch

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
    train_config_path = "config/train_config.json"
    resume_path = None # or path
    checkpoint_path = "checkpoints/best_model.pt"
    scalars_path = "outputs/exp.json"
    main(train_config_path, resume_path, checkpoint_path, scalars_path)