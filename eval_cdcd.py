# Evaluate test set from CDCD list
# Evaluation metrics: beat coverage score, beat hit score, and FAD
import torch
import argparse
import glob
import os
import random
import librosa
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from librosa.beat import beat_track
from diffusers import DiffusionPipeline
from scipy.io.wavfile import write
from PIL import Image
from tqdm.auto import tqdm
from frechet_audio_distance import FrechetAudioDistance


def beat_detect(x, sr=22050):
    onsets = librosa.onset.onset_detect(x, sr=sr, wait=1, delta=0.2, pre_avg=1, post_avg=1, post_max=1, units='time')
    n = np.ceil( len(x) / sr)
    beats = [0] * int(n)
    for time in onsets:
        beats[int(np.trunc(time))] = 1
    return beats


def beat_scores(gt, syn):
    assert len(gt) == len(syn)
    total_beats = sum(gt)
    cover_beats = sum(syn)

    hit_beats = 0
    for i in range(len(gt)):
        if gt[i] == 1 and gt[i] == syn[i]:
            hit_beats += 1
    return cover_beats/total_beats, hit_beats/total_beats


def main(args):
    # Generate Audio
    if args.gen_audio:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device)
        model_id = r"/host_data/van/edge_aistpp/modelsv2/all_01"
        encode_id = r"/host_data/van/edge_aistpp/test/concat/normalized_all_test_data_01.pkl"

    total_cover_score = 0
    total_hit_score = 0
    audio_files = [line.rstrip() for line in open('CDCD_aist.txt')]
    for audio_file in tqdm(audio_files):
        audio_file = audio_file[:-4]

        if args.gen_audio:
            #start = time.time()
            encodings = pickle.load(open(encode_id, "rb"))
            encoding = encodings[audio_file]
            print(np.array(encoding).shape)
            encoding = np.array(encoding).reshape(1, 150, 226)
            encoding = torch.Tensor(encoding).to(device)
    
            audio_diffusion = DiffusionPipeline.from_pretrained(model_id).to(device)
            mel = audio_diffusion.mel
            sample_rate = mel.get_sample_rate()

            seed = 2391504374279719  
            generator.manual_seed(seed)
            output = audio_diffusion(generator=generator, eta=0, encoding=encoding)
            image = output.images[0]
            audio = output.audios[0, 0] 

            # 64 x 64 can only output 2s so we outpaint
            if args.outpaint:
                overlap_secs = 0  
                start_step = 0  
                overlap_samples = overlap_secs * sample_rate
                track = audio
                for variation in range(3):
                    output = audio_diffusion(raw_audio=audio[-overlap_samples:], start_step=start_step, mask_start_secs=overlap_secs, eta=0, encoding=encoding)
                    audio2 = output.audios[0, 0]
                    track = np.concatenate([track, audio2[overlap_samples:]])
                    audio = audio2
                write('{}{}.wav'.format(args.output_dir, audio_file), sample_rate, track)
            else:
                write('{}{}.wav'.format(args.output_dir, audio_file), sample_rate, audio)
                #end = time.time()
                #print(end - start)
    
        else:
            # Beat Evaluation (Librosa)
            music, sr = librosa.load('{}{}.wav'.format(args.input_dir, audio_file))
            gt_beats = beat_detect(music)
            generated_audio, sr = librosa.load('{}{}.wav'.format(args.output_dir, audio_file))
            syn_beats = beat_detect(generated_audio)

            score_cover, score_hit = beat_scores(gt_beats, syn_beats)
            total_cover_score += score_cover
            total_hit_score += score_hit

    if not args.gen_audio:
        print("Score Summary for cover and hit: ", total_cover_score/len(audio_files), total_hit_score/len(audio_files))
        frechet = FrechetAudioDistance(model_name="vggish", use_pca=False, use_activation=False, verbose=False)
        fad_score = frechet.score(args.input_dir, args.output_dir)
        print("FAD: ", fad_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Dataset")
    parser.add_argument("--input_dir", type=str, default=r"/host_data/van/edge_aistpp/test/wavs_sliced/")
    parser.add_argument("--output_dir", type=str, default=r"/host_data/van/edge_aistpp/outputv2/all_01/normalized/") 
    parser.add_argument("--gen_audio", type=bool, default=False)
    parser.add_argument("--outpaint", type=bool, default=False)
    args = parser.parse_args()

    main(args)

    
