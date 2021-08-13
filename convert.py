import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter

import scipy.io.wavfile as sw
from dataset import ConversionDataset
from model import VQW2V_RNNDecoder
from tqdm import tqdm

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(convert_config_path, checkpoint_path, results_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(convert_config_path, 'r') as f:
        cfg = json.load(f)

    fix_seed(cfg['seed'])

    ds = ConversionDataset(
        root=cfg['dataset']['folder_in_archive'],
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
    outputs = []
    #meter = pyloudnorm.Meter(cfg['dataset']['sr'])
    for batch_idx, batch in enumerate(dl):
        out = model.conversion_step(batch, batch_idx)
        batch_size = len(out['cv'])
        for i in range(batch_size):
            #ref_loudness = meter.integrated_loudness(batch['audio'][j].cpu().detach().numpy())  
            converted_audio = out['cv'][i].numpy()
            #output_loudness = meter.integrated_loudness(converted_audio)
            #output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
            # save
            sw.write(filename=out['converted_audio_path'][i], rate=cfg['dataset']['sr'], data=converted_audio)
        
        del out['cv']
        outputs.append(out)
        del batch

    result = model.conversion_epoch_end(outputs)
    with open(results_path, 'w') as f:
        json.dump(result['logs'], f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-c', '-config')

    parser.add_argument('-path', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-p', '-path')

    parser.add_argument('-res', help="Path to the file of the evaluation results ", type=str, required=True)
    parser.add_argument('-r', '-res')

    args = parser.parse_args()

    # default
    # * args. = "config/convert_config.json"
    # * args.path = "checkpoints/best_model.pt"
    # * args.res = ""

    main(args.config, args.path, args.res)