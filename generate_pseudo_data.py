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

def get_save_filepath(path):
    par = Path(path).parent
    filename = Path(path)
    converted_audio_path = str(par / "wav" / f"{filename}.wav")
    representation_path = str(par / "rep" / f"{filename}.npy")
    return converted_audio_path, representation_path
    

def main(convert_config_path, checkpoint_path, outdir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(convert_config_path, 'r') as f:
        cfg = json.load(f)

    fix_seed(cfg['seed'])

    ds = ConversionDataset(
        root=cfg['dataset']['folder_in_archive'],
        outdir=outdir,
        synthesis_list_path=cfg['dataset']['synthesis_list_path'],
        sr=cfg['dataset']['sr']
    )
    dl = DataLoader(ds, batch_size=cfg['dataset']['batch_size'], num_workers=cfg['dataset']['n_workers'])

    model = VQW2V_RNNDecoder(cfg['encoder'], cfg['decoder'], device)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)

    model.eval()
    # validation phase
    meter = pyloudnorm.Meter(cfg['dataset']['sr'])
    for batch_idx, batch in enumerate(tqdm(dl)):
        out = model.conversion_step(batch, batch_idx, rep=True)
        batch_size = len(out['cv'])
        for i in range(batch_size):
            ref_loudness = meter.integrated_loudness(batch['source_audio'][i].cpu().detach().numpy())
            converted_audio = out['cv'][i].numpy()
            converted_audio = converted_audio / np.abs(converted_audio).max() * 0.999
            output_loudness = meter.integrated_loudness(converted_audio)
            converted_audio = pyloudnorm.normalize.loudness(converted_audio, output_loudness, ref_loudness)

            converted_audio_path, representation_path = get_save_filepath(out['converted_audio_path'][i])
            
            # save pseudo speech data
            sw.write(filename=converted_audio_path, rate=cfg['dataset']['sr'], data=converted_audio)
        
            rep = out['representation'][i].numpy()
            # save intermediate representation
            np.save(representation_path, rep)

        del out
        del batch


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