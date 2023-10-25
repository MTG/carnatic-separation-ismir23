import numpy as np

def GlobalSDR(references, separations):
    """ Global SDR """
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - separations), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)