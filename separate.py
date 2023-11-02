import os
import tqdm
import json
import math
import argparse

import numpy as np
import soundfile as sf
import tensorflow as tf

from config import Config as UnetConfig
from model import DiffWave
from model.vad import VAD
from model.clustering import get_mask
from utils.signal_processing import (
    compute_stft,
    compute_signal_from_stft,
    next_power_of_2,
    get_overlap_window,
)


def main(args):

    # Activate CUDA if GPU id is given
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model_path = os.path.join(".", "ckpt", args.model_name, args.model_name+"_BEST-SDR.ckpt-1")
    with open(os.path.join(".", "ckpt", args.model_name + ".json")) as f:
        unet_config = UnetConfig.load(json.load(f))

    diffwave = DiffWave(unet_config.model)
    diffwave.restore(model_path).expect_partial()
    mixture = tf.io.read_file(args.input_signal)
    mixture, _ =  tf.audio.decode_wav(mixture, desired_channels=1)
    mixture = tf.squeeze(mixture, axis=-1) / tf.reduce_max(mixture)

    TRIMS = args.batch_size
    output_voc = np.zeros(mixture.shape)
    hopsized_batch = int((TRIMS*22050) / 2)
    runs = math.floor(mixture.shape[0] / hopsized_batch)
    trim_low = 0
    for trim in tqdm.tqdm(np.arange((runs*2)-1)):
        trim_high = int(trim_low + (hopsized_batch*2))

        # Get input mixture spectrogram
        mix_trim = mixture[trim_low:trim_high]
        mix_mag, mix_phase = compute_stft(mix_trim[None], unet_config)
        new_len = next_power_of_2(mix_mag.shape[1])
        mix_mag_trim = mix_mag[:, :new_len, :]
        mix_phase_trim = mix_phase[:, :new_len, :]

        # Get and stack cold diffusion steps
        diff_feat = diffwave(mix_mag_trim, mode="train")
        diff_feat = tf.transpose(diff_feat, [1, 0, 2, 3])
        diff_feat_t = tf.squeeze(tf.reshape(diff_feat, [1, 8, diff_feat.shape[-2]*diff_feat.shape[-1]]), axis=0).numpy()

        # Normalize features, all energy curves having same range
        normalized_feat = []
        for j in np.arange(diff_feat_t.shape[1]):
            normalized_curve = diff_feat_t[:, j] / (np.max(np.abs(diff_feat_t[:, j]))+1e-6)
            normalized_feat.append(normalized_curve)
        normalized_feat = np.array(normalized_feat, dtype=np.float32)

        # Compute mask using unsupervised clustering and reshape to magnitude spec shape
        mask = get_mask(normalized_feat, args.clusters, args.scheduler)
        mask = tf.reshape(mask, mix_mag_trim.shape)

        # Getting last step of computed features and applying mask
        diff_feat_t = tf.reshape(diff_feat_t[-1, :], mix_mag_trim.shape)
        output_signal = tf.math.multiply(diff_feat_t, mask)

        # Silence unvoiced regions
        output_signal = compute_signal_from_stft(output_signal, mix_phase_trim, unet_config)
        pred_audio = tf.squeeze(output_signal, axis=0).numpy()
        vad = VAD(pred_audio, sr=22050, nFFT=512, win_length=0.025, hop_length=0.01, theshold=0.99)
        if np.sum(vad) / len(vad) < 0.25:
            pred_audio = np.zeros(pred_audio.shape)

        # Get boundary
        boundary = None
        boundary = "start" if trim == 0 else None
        boundary = "end" if trim == runs-2 else None

        placehold_voc = np.zeros(output_voc.shape)
        placehold_voc[trim_low:trim_low+pred_audio.shape[0]] = pred_audio * get_overlap_window(pred_audio, boundary=boundary)
        output_voc += placehold_voc
        trim_low += pred_audio.shape[0] // 2

    output_voc = output_voc * (np.max(np.abs(mixture.numpy())) / (np.max(np.abs(output_voc))+1e-6))
    
    # Building intuitive filename with model config
    filefolder = os.path.join(args.input_signal.split("/")[:-1])
    filename = args.input_signal.split("/")[-1].split(".")[:-1]
    filename = filename[0] if len(filename) == 1 else ".".join(filename)
    filename = filename + "_" + str(args.clusters) + "_" + str(args.scheduler) + "pred_voc"
    sf.write(
        os.path.join(filefolder, filename + ".wav"),
        output_voc,
        22050) # Writing to file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='saraga-8')
    parser.add_argument('--input_signal', default=None, type=str)
    parser.add_argument('--batch-size', default=3)
    parser.add_argument('--clusters', default=4, type=int)
    parser.add_argument('--scheduler', default=3., type=float)
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()
    main(args)
