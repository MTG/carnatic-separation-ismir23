import os
import argparse
import json
import tqdm

import numpy as np
import tensorflow as tf

from config import Config as UnetConfig
from dataset import SARAGA
from model import DiffWave

from utils.separation_eval import GlobalSDR
from utils.signal_processing import check_shape_3d

import warnings
warnings.filterwarnings('ignore')

epsilon = 1e-6

class Trainer:
    """WaveGrad trainer.
    """
    def __init__(self, model, saraga, config, data_dir):
        """Initializer.
        Args:
            model: DiffWave, diffwave model.
            saraga: Saraga, saraga dataset
                which provides already batched and normalized speech dataset.
            config: Config, unified configurations.
        """
        self.model = model
        self.saraga = saraga
        self.config = config
        self.data_dir = data_dir

        self.split = config.train.split // config.data.batch
        self.trainset = self.saraga.dataset().take(self.split) \
            .shuffle(config.train.bufsiz) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        self.testset = self.saraga.test_dataset() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        self.optim = tf.keras.optimizers.Adam(
            config.train.lr(),
            config.train.beta1,
            config.train.beta2,
            config.train.eps)

        self.eval_intval = config.train.eval_intval // config.data.batch
        self.ckpt_intval = config.train.ckpt_intval // config.data.batch

        self.train_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.alpha_bar = np.linspace(1, 0, config.model.iter + 1)

    @staticmethod
    def tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def compute_loss(self, mixture, vocals): #, accomp):
        """Compute loss for noise estimation.
        Args:
            signal: tf.Tensor, [B, T], raw audio signal mixture.
            signal: tf.Tensor, [B, T], raw audio signal vocals.
        Returns:
            loss: tf.Tensor, [], L1-loss between noise and estimation.
        """
        bsize = tf.shape(vocals)[0]
        # [B]
        timesteps = tf.random.uniform(
            [bsize], 1, self.config.model.iter + 1, dtype=tf.int32)

        # [B]
        noise_level = tf.gather(self.alpha_bar, timesteps)
        noise_level_next = tf.gather(self.alpha_bar, timesteps - 1)

        # [B, T], [B, T]
        noised = self.model.diffusion(mixture, vocals, noise_level)
        noised_next = self.model.diffusion(mixture, vocals, noise_level_next)
        # [B, T]
        est = self.model.pred_noise(noised, timesteps)
        # []
        l1_loss = tf.reduce_mean(tf.abs(est - noised_next))
        return l1_loss


    def train(self, step=0):
        """Train wavegrad.
        Args:
            step: int, starting step.
            ir_unit: int, log ir units.
        """
        count = 0
        best_SDR = 0
        best_step = 0
        less_loss = 1000
        less_train_loss = 1000
        # Start training
        for _ in tqdm.trange(step // self.split, self.config.train.epoch):
            train_loss = []
            with tqdm.tqdm(total=self.split, leave=False) as pbar:
                for mixture, vocal in self.trainset:
                    with tf.GradientTape() as tape:
                        tape.watch(self.model.trainable_variables)
                        loss = self.compute_loss(mixture, vocal)
                        train_loss.append(loss)

                    grad = tape.gradient(loss, self.model.trainable_variables)
                    self.optim.apply_gradients(
                        zip(grad, self.model.trainable_variables))

                    norm = tf.reduce_mean([tf.norm(g) for g in grad])
                    del grad

                    step += 1
                    pbar.update()
                    pbar.set_postfix(
                        {'loss': loss.numpy().item(),
                         'step': step,
                         'grad': norm.numpy().item()})

                    if step % self.ckpt_intval == 0:
                        self.model.write(
                            '{}.ckpt'.format(self.ckpt_path),
                            self.optim)

            train_loss = sum(train_loss) / len(train_loss)
            print('\nTrain loss:', str(round(train_loss.numpy(), 5)))
            loss = []
            for mixture, vocal in self.testset:
                actual_loss = self.compute_loss(mixture, vocal)
                loss.append(actual_loss.numpy().item())

            del mixture, vocal
            loss = sum(loss) / len(loss)
            print('Eval loss:', str(round(loss, 5)))
            if loss <= less_loss:
                if train_loss <= less_train_loss:
                    print('Saving best new model given loss values!')
                    self.model.write('{}_BEST-LOSS.ckpt'.format(self.ckpt_path),self.optim)
                    less_loss = loss
                    less_train_loss = train_loss

            with self.test_log.as_default():
                if count%1 == 0:
                    best_SDR, best_step = self.eval(best_SDR, best_step, step)
            count += 1


    def eval(self, best_SDR, best_step, step):
        """Generate evaluation purpose audio.
        Returns:
            speech: np.ndarray, [T], ground truth.
            pred: np.ndarray, [T], predicted.
            ir: List[np.ndarray], config.model.iter x [T],
                intermediate representations.
        """
        # [T]
        voc_sdr = []
        for mixture, vocals in tqdm.tqdm(saraga.validation().take(300)):
            if np.max(tf.squeeze(mixture, axis=0).numpy())>0:
                if np.max(tf.squeeze(vocals, axis=0).numpy())>0:
                    mix_mag, _ = self.compute_stft(mixture)
                    _, voc_phase = self.compute_stft(vocals)

                    pred = self.model(mix_mag)
                    pred = self.compute_signal_from_stft(pred, voc_phase)
                    mixture = mixture[:, :pred.shape[1]]
                    vocals = vocals[:, :pred.shape[1]]
                    pred = tf.transpose(pred, [1, 0]).numpy()
                    vocals = tf.transpose(vocals, [1, 0]).numpy()

                    ref = np.array([vocals])
                    est = np.array([pred])

                    scores = GlobalSDR(ref, est)
                    voc_sdr.append(scores[0])

        print('Median SDR:', np.median(voc_sdr))
        print('Best model:', best_SDR)
        if np.median(voc_sdr) > best_SDR:
            print('Saving best new model with SDR:', np.median(voc_sdr))
            self.model.write('{}_BEST-SDR.ckpt'.format(self.ckpt_path),self.optim)
            best_SDR = np.median(voc_sdr)
        return best_SDR, best_step

    def compute_stft(self, signal):
        signal_stft = check_shape_3d(
            check_shape_3d(
                tf.signal.stft(
                    signal,
                    frame_length=self.config.model.win,
                    frame_step=self.config.model.hop,
                    fft_length=self.config.model.win,
                    window_fn=tf.signal.hann_window), 1), 2)
        mag = tf.abs(signal_stft)
        phase = tf.math.angle(signal_stft)
        return mag, phase

    def compute_signal_from_stft(self, spec, phase):
        polar_spec = tf.complex(tf.multiply(spec, tf.math.cos(phase)), tf.zeros(spec.shape)) + \
            tf.multiply(tf.complex(spec, tf.zeros(spec.shape)), tf.complex(tf.zeros(phase.shape), tf.math.sin(phase)))
        return tf.signal.inverse_stft(
            polar_spec,
            frame_length=self.config.model.win,
            frame_step=self.config.model.hop,
            window_fn=tf.signal.inverse_stft_window_fn(
                self.config.model.hop,
                forward_window_fn=tf.signal.hann_window))

    @staticmethod
    def load_audio(paths):
        mixture = tf.io.read_file(paths[0])
        vocals = tf.io.read_file(paths[1])
        mixture_audio, _ = tf.audio.decode_wav(mixture, desired_channels=1)
        vocal_audio, _ = tf.audio.decode_wav(vocals, desired_channels=1)
        return tf.squeeze(mixture_audio, axis=-1), tf.squeeze(vocal_audio, axis=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-step', default=0, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()

    # Activate CUDA if GPU id is given
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    config = UnetConfig()

    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = UnetConfig.load(json.load(f))

    log_path = os.path.join(config.train.log, config.train.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    sounds_path = os.path.join(config.train.sounds, config.train.name)
    if not os.path.exists(sounds_path):
        os.makedirs(sounds_path)

    saraga = SARAGA(config.data, data_dir=args.data_dir)
    diffwave = DiffWave(config.model)
    trainer = Trainer(diffwave, saraga, config, data_dir=args.data_dir)

    if args.load_step > 0:
        super_path = os.path.join(config.train.ckpt, config.train.name)
        ckpt_path = os.path.join(super_path, '{}.ckpt-1'.format(config.train.name))
        print('[*] load checkpoint: ' + ckpt_path)
        trainer.model.restore(ckpt_path, trainer.optim)
        print("Loaded!")

    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    trainer.train(args.load_step)
