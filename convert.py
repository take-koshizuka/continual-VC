import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ConversionDataset
from model import VQW2V_RNNDecoder

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

    ds = ConversionDataset(
        root=cfg['dataset']['folder_in_archive'],
        data_list_path=cfg['dataset']['synthesis_list_path'],
        sr=cfg['dataset']['sr'],
        bits=cfg['dataset']['bits'],
    )

    dl = DataLoader(ds, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['n_workers'])

    model = VQW2V_RNNDecoder(cfg['encoder'], cfg['decoder'], device)

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)

    model.encoder.eval()

    # training phase
    for batch_idx, batch in enumerate(dl):
        out = model._step(batch, batch_idx)
        del batch

        
if __name__ == '__main__':
    train_config_path = "config/train_config.json"
    resume_path = None # or path
    checkpoint_path = "checkpoints/best_model.pt"
    scalars_path = "outputs/exp.json"
    main(train_config_path, resume_path, checkpoint_path, scalars_path)