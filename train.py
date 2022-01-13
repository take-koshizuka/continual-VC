import random
import json
import numpy as np
from numpy.random.mtrand import random_sample
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import WavDataset
from model import VQW2V_RNNDecoder
from utils import EarlyStopping, get_metadata
from tqdm import tqdm
from pathlib import Path
import argparse

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

def main(train_config_path, checkpoint_dir, resume_path=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(train_config_path, 'r') as f:
        cfg = json.load(f)
    fix_seed(cfg['seed'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    train_metadata, val_metadata = get_metadata(cfg['path_list_dir'], cfg["speakers"], 
                                        train_size=cfg['train_size'], val_size=cfg['val_size'], random_split=cfg['random_split'])

    tr_ds = WavDataset(
        root=cfg['dataset']['folder_in_archive'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        metadata=train_metadata
    )

    va_ds = WavDataset(
        root=cfg['dataset']['folder_in_archive'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        metadata=val_metadata
    )

    tr_dl = DataLoader(tr_ds,
            batch_size=cfg['dataset']['batch_size'],
            shuffle=True,
            drop_last=True)
    
    va_bs = len(va_ds) if len(va_ds) <= 64 else 25
    va_dl = DataLoader(va_ds,
            batch_size=va_bs,
            drop_last=True)

    model = VQW2V_RNNDecoder(cfg['encoder'], cfg['decoder'], device)
    model.to(device)

    optimizer = optim.Adam(
                    model.decoder.parameters(),
                    lr=cfg['optim']['lr'],
                    betas=(float(cfg['optim']['beta_0']), float(cfg['optim']['beta_1']))
                )

    scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=cfg['scheduler']['milestones'],
                    gamma=cfg['scheduler']['gamma']
                )

    early_stopping = EarlyStopping('avg_loss', 'min')

    init_epochs = 1
    max_epochs = cfg['epochs']

    if AMP:
        model.decoder, optimizer = amp.initialize(model.decoder, optimizer, opt_level="O1")

    if not cfg['decoder_checkpoint'] == "":
        checkpoint = torch.load(cfg['decoder_checkpoint'], map_location=lambda storage, loc: storage)
        model.load_model(checkpoint, amp=False)
        model.to(device)

    if not resume_path == "":
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_model(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        init_epochs = checkpoint['epochs']
        model.to(device)

    model.encoder.eval()
    writer = SummaryWriter()
    records = {}
    
    for i in tqdm(range(init_epochs, max_epochs + 1)):
        # training phase
        for batch_idx, train_batch in enumerate(tqdm(tr_dl, leave=False)):
            optimizer.zero_grad()
            out = model.training_step(train_batch, batch_idx)
            if AMP:
                with amp.scale_loss(out['loss'], optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            else:
                out['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1)

            optimizer.step()
            del train_batch
        scheduler.step()

        # validation phase
        outputs = []
        for batch_idx, eval_batch in enumerate(tqdm(va_dl, leave=False)):
            out = model.validation_step(eval_batch, batch_idx)
            outputs.append(out)
            del eval_batch

        val_result = model.validation_epoch_end(outputs)
        writer.add_scalars('data/loss', val_result['log'])
        records[f'epoch {i}'] = val_result['log']

        with open(str(checkpoint_dir / f"records_elapse.json"), "w") as f:
            json.dump(records, f, indent=4)
        
        # early_stopping
        if early_stopping.judge(val_result):
            early_stopping.update(val_result)
            state_dict = model.state_dict(optimizer, scheduler)
            state_dict['epochs'] = i
            early_stopping.best_state = state_dict

        if i % cfg['checkpoint_period'] == 0:
            state_dict = model.state_dict(optimizer, scheduler)
            state_dict['epochs'] = i
            torch.save(state_dict, str(checkpoint_dir / f"model-{i}.pt"))
    
    best_state = early_stopping.best_state
    torch.save(best_state, str(checkpoint_dir / "best-model.pt"))
    with open(str(checkpoint_dir / "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    
    with open(str(checkpoint_dir / f"records-{init_epochs}-{max_epochs}.json"), "w") as f:
        json.dump(records, f, indent=4)
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for training.", type=str, required=True)
    parser.add_argument('-dir', '-d', help="Path to the directory where the checkpoint of the model is stored.", type=str, required=True)
    parser.add_argument('-resume', '-r', help="Path to the checkpoint of the model you want to resume training.", type=str, default="")

    args = parser.parse_args()

    ## example
    # args.config = "config/train_fine.json"
    # args.dir = "checkpoints/fine"
    # args.resume = "checkpoints/baseline493/best-model.pt"
    ##

    main(args.config, Path(args.dir), args.resume)