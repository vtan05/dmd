# Normalize loudness of generated music to match ground truth
import argparse
import glob
import os
from tqdm.auto import tqdm
from pydub import AudioSegment, effects


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def main(args):
    audio_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    for audio_file in tqdm(audio_files):
        audio_file = (os.path.basename(audio_file).split('/')[-1][:-4])
        sound = AudioSegment.from_file('{}{}.wav'.format(args.input_dir, audio_file), format="wav", frame_rate=22050)
        five_seconds = 5 * 1000
        first_5_seconds = sound[:five_seconds]
        normalized_sound = match_target_amplitude(first_5_seconds, -20.0)
        normalized_sound = normalized_sound + 8
        normalized_sound.export('{}{}.wav'.format(args.output_dir, audio_file), format="wav")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Normalize Volume")
    parser.add_argument("--input_dir", type=str, default="wav/")
    parser.add_argument("--output_dir", type=str, default="wav_norm/")
    args = parser.parse_args()

    main(args)