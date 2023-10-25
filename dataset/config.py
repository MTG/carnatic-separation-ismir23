import tensorflow as tf

class Config:
    """Configuration for dataset construction.
    """
    def __init__(self):
        # audio config
        self.sr = 22050

        # stft
        self.hop = 256
        self.win = 1024
        self.fft = self.win
        self.win_fn = 'hann'

        # mel-scale filter bank
        self.mel = 80
        self.fmin = 0
        self.fmax = 8000

        self.eps = 1e-5

        # sample size
        self.frames = (self.hop + 6) * 128  # 16384
        self.batch = 8

        self.eval_tracks = [
            'kailasapathe', 'ragam_tanam_pallavi', 'gopi_gopala_bala', 'ananda_sagara'
        ]  # four manually selected tracks for validation

    def window_fn(self):
        """Return window generator.
        Returns:
            Callable, window function of tf.signal
                , which corresponds to self.win_fn.
        """
        mapper = {
            'hann': tf.signal.hann_window,
            'hamming': tf.signal.hamming_window
        }
        if self.win_fn in mapper:
            return mapper[self.win_fn]
        
        raise ValueError('invalid window function: ' + self.win_fn)
