import numpy as np


def ece(y, y_probas, n_bins=10):
    """Expected calibration error.

    Parameters
    ----------
    y : NumPy array of shape (n_samples, n_classes) or (n_samples,)
        True labels. Either as label vector (1-D) or as one-hot encoded ground 
        truth array (2-D).
    y_probas : NumPy array of shape (n_samples, n_classes) or (n_samples,)
        Predicted probabilities.

    Returns
    -------
    ece : float
        ECE score.
    """
    if y_probas.ndim == 2:
        max_indices = y_probas.argmax(axis=-1)
        if y.ndim == 2:
            indices = np.expand_dims(max_indices, axis=-1)
            y = np.take_along_axis(y, indices, axis=-1).squeeze(axis=-1)
        elif not np.array_equal(y, y.astype(bool)):
            n_classes = y_probas.shape[1]
            assert set(y).issubset(range(n_classes))
            y = np.array([yi == i for yi, i in zip(y, max_indices)], dtype=bool)
        indices = np.expand_dims(max_indices, axis=-1)
        y_probas = np.take_along_axis(y_probas, indices, axis=-1).squeeze(axis=-1)
    else:
        assert y.ndim == 1 and np.array_equal(y, y.astype(bool))
        
    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(y_probas, bins, right=False)  # Assuming no probabilities equal 1.0

    _ece = 0
    for i in range(1, n_bins+1):
        bin_mask = bin_indices == i
        if bin_mask.sum() == 0:
            continue
        y_probas_bin = y_probas[bin_mask]
        conf = np.mean(y_probas_bin)
        y_bin = y[bin_mask]     
        acc = np.mean(y_bin)
        _ece += len(y_bin) * np.abs(acc-conf)

    return _ece / len(y_probas)


def sce(y_one_hot, y_probas, n_bins=10):
    """Static calibration error."""
    _sce = 0
    for yp, y in zip(y_probas.T, y_one_hot.T):
        _sce += ece(y, yp, n_bins)
    return _sce / y_probas.shape[1]
