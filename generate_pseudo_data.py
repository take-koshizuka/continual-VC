import random
import json
import argparse
import gc
import os
from tqdm import tqdm
from pathlib import Path
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

def get_save_filepath(path):
    par = Path(path).parent
    filename = Path(path).name
    converted_audio_path = str(par / "wav" / f"{filename}.npy")
    representation_path = str(par / "rep" / f"{filename}.npy")
    return converted_audio_path, representation_path
    

def main(convert_config_path, checkpoint_path, outdir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav_dir = Path(outdir) / "wav"
    rep_dir = Path(outdir) / "rep"
    wav_dir.mkdir(exist_ok=True, parents=True)
    rep_dir.mkdir(exist_ok=True, parents=True)

    with open(convert_config_path, 'r') as f:
        cfg = json.load(f)

    fix_seed(cfg['seed'])

    ds = ConversionDataset(
        root=cfg['dataset']['folder_in_archive'],
        outdir=outdir,
        synthesis_list_path=cfg['dataset']['synthesis_list_path'],
        sr=cfg['dataset']['sr'],
        unlabeled=True
    )
    dl = DataLoader(ds, batch_size=1)

    model = VQW2V_RNNDecoder(cfg['encoder'], cfg['decoder'], device)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint, amp=False)
    model.to(device)

    model.eval()

    for batch_idx, batch in enumerate(tqdm(dl)):
        converted_audio_path, representation_path = get_save_filepath(batch['converted_audio_path'][0])
        if os.path.isfile(converted_audio_path):
            continue
        
        out = model.conversion_step(batch, batch_idx, rep=True)
        converted_audio = out['cv'].numpy()
        converted_audio = converted_audio / np.abs(converted_audio).max() * (1.0 - 1e-8)
        converted_audio_path, representation_path = get_save_filepath(out['converted_audio_path'])

        np.save(converted_audio_path, converted_audio)
        rep = out['representation'].numpy()
        # save intermediate representation
        np.save(representation_path, rep)
        
        del converted_audio
        del rep
        del out
        del batch
        gc.collect()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-outdir', '-d', help="Path to the output directory of converted speeches", type=str, required=True)

    args = parser.parse_args()

    ## example
    # args.config = "config/convert_pre.json"
    # args.path = "checkpoints/baseline493/best-model.pt"
    # args.outdir = "pseudo_speech"
    ##

    main(args.config, args.path, args.outdir)
