import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

import pyloudnorm
import scipy.io.wavfile as sw
from dataset import ConversionDataset
from model import VQW2V_RNNDecoder
from tqdm import tqdm
from pathlib import Path

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(convert_config_path, checkpoint_path, outdir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(outdir).mkdir(exist_ok=True, parents=True)
    with open(convert_config_path, 'r') as f:
        cfg = json.load(f)

    fix_seed(cfg['seed'])

    ds = ConversionDataset(
        root=cfg['dataset']['folder_in_archive'],
        outdir=outdir,
        synthesis_list_path=cfg['dataset']['synthesis_list_path'],
        sr=cfg['dataset']['sr']
    )
    
    dl = DataLoader(ds, batch_size=1)

    model = VQW2V_RNNDecoder(cfg['encoder'], cfg['decoder'], device)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint, amp=False)
    model.to(device)

    model.eval()
    # validation phase
    outputs = []
    meter = pyloudnorm.Meter(cfg['dataset']['sr'])
    for batch_idx, batch in enumerate(tqdm(dl)):
        out = dict(
            converted_audio_path=batch['source_audio_path'][0],
            target_audio=batch['target_audio'][0],
            utterance=batch['utterance'][0]
        )
        outputs.append(out)
        del batch

    result = model.conversion_epoch_end(outputs)
    results_path = str(Path(outdir) / "results.json")
    with open(results_path, 'w') as f:
        json.dump(result['logs'], f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-outdir', '-d', help="Path to the output directory of converted speeches", type=str, required=True)

    args = parser.parse_args()

    # default
    # * args. = "config/convert_config.json"
    # * args.path = "checkpoints/best_model.pt"
    # * args.res = ""

    main(args.config, args.path, args.outdir)