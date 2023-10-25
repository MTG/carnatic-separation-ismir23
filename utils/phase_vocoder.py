import numpy as np
import tensorflow as tf


def phase_vocoder(D, hop_len=256, rate=0.8):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of `rate`.
    Based on implementation provided by:
      https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#phase_vocoder
    :param D: tf.complex64([num_frames, num_bins]): the STFT tensor
    :param hop_len: float: the hop length param of the STFT
    :param rate: float > 0: the speed-up factor
    :return: D_stretched: tf.complex64([num_frames, num_bins]): the stretched STFT tensor
    """
    # get shape
    sh = tf.shape(D, name="STFT_shape")
    frames = sh[0]
    fbins = sh[1]

    # time steps range
    t = tf.range(0.0, tf.cast(frames, tf.float32), rate, dtype=tf.float32, name="time_steps")

    # Expected phase advance in each bin
    dphi = tf.linspace(0.0, np.pi * hop_len, fbins, name="dphi_expected_phase_advance")
    phase_acc = tf.math.angle(D[0, :], name="phase_acc_init")

    # Pad 0 columns to simplify boundary logic
    D = tf.pad(D, [(0, 2), (0, 0)], mode='CONSTANT', name="padded_STFT")

    # def fn(previous_output, current_input):
    def _pvoc_mag_and_cum_phase(previous_output, current_input):
        # unpack prev phase
        _, prev = previous_output

        # grab the two current columns of the STFT
        i = tf.cast((tf.floor(current_input) + [0, 1]), tf.int32)
        bcols = tf.gather_nd(D, [[i[0]], [i[1]]])

        # Weighting for linear magnitude interpolation
        t_dif = current_input - tf.floor(current_input)
        bmag = (1 - t_dif) * tf.abs(bcols[0, :]) + t_dif * (tf.abs(bcols[1, :]))

        # Compute phase advance
        dp = tf.math.angle(bcols[1, :]) - tf.math.angle(bcols[0, :]) - dphi
        dp = dp - 2 * np.pi * tf.round(dp / (2.0 * np.pi))

        # return linear mag, accumulated phase
        return bmag, tf.squeeze(prev + dp + dphi)

    # initializer of zeros of correct shape for mag, and phase_acc for phase
    initializer = (tf.zeros(fbins, tf.float32), phase_acc)
    mag, phase = tf.scan(_pvoc_mag_and_cum_phase, t, initializer=initializer,
                            parallel_iterations=10, back_prop=False,
                            name="pvoc_cum_phase")

    # add the original phase_acc in
    phase2 = tf.concat([tf.expand_dims(phase_acc, 0), phase], 0)[:-1, :]
    D_stretched = tf.cast(mag, tf.complex64) * tf.exp(1.j * tf.cast(phase2, tf.complex64), name="stretched_STFT")

    return D_stretched