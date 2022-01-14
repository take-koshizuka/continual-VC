import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import WavDataset, PseudoWavDataset, BalancedBatchSampler
from model import VQW2V_RNNDecoder_Replay
from utils import EarlyStopping, get_metadata
from tqdm import tqdm
from pathlib import Path
import argparse
import gc

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

    train_metadata_fine, val_metadata_fine = get_metadata(cfg['fine']['path_list_dir'], cfg["fine"]["speakers"], train_size=cfg["fine"]["train_size"], 
                                                            val_size=cfg["fine"]["val_size"], random_split=cfg['random_split'])

    train_metadata_pre, val_metadata_pre = get_metadata(cfg['pre']['path_list_dir'], cfg["pre"]["speakers"], train_size=cfg["pre"]["train_size"], 
                                                            val_size=cfg["pre"]["val_size"], random_split=cfg['random_split'])

    tr_ds_fine = WavDataset(
        root=cfg['dataset']['folder_in_archive'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        metadata=train_metadata_fine
    )

    va_ds_fine = WavDataset(
        root=cfg['dataset']['folder_in_archive'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        metadata=val_metadata_fine
    )

    tr_ds_pre = PseudoWavDataset(
        root=cfg['dataset']['folder_pseudo_speech'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        metadata=train_metadata_pre
    )
    
    va_ds_pre = PseudoWavDataset(
        root=cfg['dataset']['folder_pseudo_speech'],
        sr=cfg['dataset']['sr'],
        sample_frames=cfg['dataset']['sample_frames'],
        hop_length=cfg['dataset']['hop_length'],
        metadata=val_metadata_pre
    )

    batch_sampler_fine = BalancedBatchSampler(tr_ds_fine, cfg['dataset']['batch_size_per_class_fine'])
    tr_dl_fine = DataLoader(tr_ds_fine, batch_sampler=batch_sampler_fine)

    batch_sampler_pre = BalancedBatchSampler(tr_ds_pre.labels, cfg['dataset']['batch_size_per_class_pre'])
    tr_dl_pre = DataLoader(tr_ds_pre, batch_sampler=batch_sampler_pre)

    va_dl_fine = DataLoader(va_ds_fine, batch_size=50, drop_last=False)
    va_dl_pre = DataLoader(va_ds_pre, batch_size=50, drop_last=False)
    
    tr_it_pre = iter(tr_dl_pre)
    va_it_pre = iter(va_dl_pre)
    
    model = VQW2V_RNNDecoder_Replay(cfg['encoder'], cfg['decoder'], device)
    model.to(device)

    optimizer = optim.Adam(
                    model.decoder.parameters(),
                    lr=cfg['optim']['lr'],
                    betas=(float(cfg['optim']['beta_0']), float(cfg['optim']['beta_1']))
                )

    init_epochs = 1
    max_epochs = cfg['epochs']

    scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=cfg['scheduler']['milestones'],
                    gamma=cfg['scheduler']['gamma']
                )
    
    early_stopping = EarlyStopping('avg_loss', 'min')

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
        for batch_idx, train_batch_fine in enumerate(tqdm(tr_dl_fine, leave=False)):
            try:
                train_batch_pre = next(tr_it_pre) 
            except StopIteration:
                tr_it_pre = iter(tr_dl_pre)
                train_batch_pre = next(tr_it_pre)

            optimizer.zero_grad()
            out = model.training_step(train_batch_fine, train_batch_pre, batch_idx)
            if AMP:
                with amp.scale_loss(out['loss'], optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            else:
                out['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1)

            optimizer.step()

            del train_batch_fine
            del train_batch_pre
        
            gc.collect()
            scheduler.step()
        # validation phase
        outputs = []
        for batch_idx, val_batch_fine in enumerate(tqdm(va_dl_fine, leave=False)):
            try:
                val_batch_pre = next(va_it_pre) 
            except StopIteration:
                va_it_pre = iter(va_dl_pre)
                val_batch_pre = next(va_it_pre)

            out = model.validation_step(val_batch_fine, val_batch_pre, batch_idx)
            outputs.append(out)

            del val_batch_fine
            del val_batch_pre

            gc.collect()

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
    print(f"save_checkpoint epoch:{best_state['epochs']}")
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
    # args.config = "config/train_preh.json"
    # args.dir = "checkpoints/preh"
    ##

    main(args.config, Path(args.dir), args.resume)
