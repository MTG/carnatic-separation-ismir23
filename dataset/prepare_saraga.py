import os
import argparse
import torch
import glob
import tqdm

import numpy as np
import torch.nn.functional as F
import torchaudio as T

SR = 22050


def main(args):
    concert = glob.glob(os.path.join(args.saraga_dir, '*/'))

    for i in tqdm(concert):
        songs = glob.glob(os.path.join(args.saraga_dir, i, '*/'))
        for j in tqdm.tqdm(songs):
            song_name = j.split("/")[-2]
            mixture = os.path.join(j, song_name + ".mp3.mp3")
            vocals = os.path.join(j, song_name + ".multitrack-vocal.mp3")

            if os.path.exists(mixture):
                audio_mix, sr = T.load(mixture)
                audio_voc, _ = T.load(vocals)
                resampling = T.transforms.Resample(sr, SR)
                audio_mix = resampling(audio_mix)
                audio_voc = resampling(audio_voc)
                audio_mix = torch.mean(audio_mix, dim=0).unsqueeze(0)
                audio_mix = torch.clamp(audio_mix, -1.0, 1.0)
                audio_voc = torch.mean(audio_voc, dim=0).unsqueeze(0)
                audio_voc = torch.clamp(audio_voc, -1.0, 1.0)

                actual_len = audio_voc.shape
                for trim in np.arange(actual_len[1] // (args.sample_len*SR)):
                    T.save(
                        os.path.join(
                            args.output_dir, song_name.lower().replace(" ", "_") + '_' + str(trim) + '_mixture.wav'),
                        audio_mix[:, trim*args.sample_len*SR:(trim+1)*args.sample_len*SR].cpu(),
                        sample_rate=sr,
                        bits_per_sample=16)
                    T.save(
                        os.path.join(
                            args.output_dir, song_name.lower().replace(" ", "_") + '_' + str(trim) + '_vocals.wav'),
                        audio_voc[:, trim*args.sample_len*SR:(trim+1)*args.sample_len*SR].cpu(),
                        sample_rate=sr,
                        bits_per_sample=16)
            else:
                print("no file...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saraga-dir', default=None, type=str)
    parser.add_argument('--output-dir', default=None, type=str)
    parser.add_argument('--sample-len', default=6)
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()
    main(args)
