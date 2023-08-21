# Evaluate Genre KLD on test set of CDCD list
# Model used: https://github.com/PeiChunChang/MS-SincResNet
import argparse
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import librosa
import librosa.display
import torch

from genremodel.models import *
from scipy import signal
from tqdm.auto import tqdm
from scipy.special import kl_div


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def main(args):
    
    # Load MS-SincResNet model
    MODEL_PATH = 'MS-SincResNet.tar'
    state_dict = torch.load(MODEL_PATH)
    model = MS_SincResNet()
    model.load_state_dict(state_dict['state_dict'])
    model.cuda()
    model.eval()

    audio_files = [line.rstrip() for line in open('CDCD_aist.txt')]
    kl_array = []
    for audio_file in tqdm(audio_files):
        _, data = scipy.io.wavfile.read('{}{}'.format(args.input_dir, audio_file))
        data = signal.resample(data, 16000 * 30)
        data = data[24000:72000]

        data = torch.from_numpy(data).float()
        data.unsqueeze_(dim=0)
        data.unsqueeze_(dim=0)
        data = data.cuda()
        gt, _, _, _ = model(data)
        gt = gt.detach().cpu().numpy().tolist()[0]
        gt = NormalizeData(gt)
        
        _, data = scipy.io.wavfile.read('{}{}'.format(args.output_dir, audio_file))
        data = signal.resample(data, 16000 * 30)
        data = data[24000:72000]

        data = torch.from_numpy(data).float()
        data.unsqueeze_(dim=0)
        data.unsqueeze_(dim=0)
        data = data.cuda()
        gen, _, _, _ = model(data)
        gen = gen.detach().cpu().numpy().tolist()[0]
        gen = NormalizeData(gen)
    
        output = sum(kl_div(np.array(gt), np.array(gen)))

        kl_array.append(output)
    print(np.mean(ma.masked_invalid(kl_array)))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genre Classifier")
    parser.add_argument("--input_dir", type=str, default=r"/host_data/van/edge_aistpp/test/wavs_sliced/")
    parser.add_argument("--output_dir", type=str, default=r"/host_data/van/edge_aistpp/outputv2/all_01/normalized/") 
    args = parser.parse_args()

    main(args)