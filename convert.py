import random
import json
import numpy as np
import torch
import pandas as pd
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
        out = model.conversion_step(batch, batch_idx)
        ref_loudness = meter.integrated_loudness(batch['source_audio'][0].cpu().detach().numpy())  
        converted_audio = out['cv'].numpy()
        converted_audio = converted_audio / np.abs(converted_audio).max() * 0.999
        output_loudness = meter.integrated_loudness(converted_audio)
        converted_audio = pyloudnorm.normalize.loudness(converted_audio, output_loudness, ref_loudness)
        # save
        sw.write(filename=out['converted_audio_path'], rate=cfg['dataset']['sr'], data=converted_audio)
        
        del out['cv']
        outputs.append(out)
        del batch

    result = model.conversion_epoch_end(outputs)
    results_path = str(Path(outdir) / "results.json")
    with open(results_path, 'w') as f:
        json.dump(result['logs'], f, indent=4)
    
    all_results_path = str(Path(outdir) / "all_results.csv")
    df = pd.DataFrame(result['all_logs'])
    df.to_csv(all_results_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-outdir', '-d', help="Path to the output directory of converted speeches", type=str, required=True)

    args = parser.parse_args()

    ## example
    # args.config = "config/convert_fine.json"
    # args.path = "checkpoints/fine/best-model.pt"
    # args.outdir = "outputs/fine"
    ##

    main(args.config, args.path, args.outdir)