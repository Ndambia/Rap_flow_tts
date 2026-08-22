def make_delta_schedule(delta_start=0.1, delta_end=0.001, n_bins=8, n_epochs=1400):
    """Returns a function epoch -> delta_t."""
    bin_len = max(1, n_epochs // n_bins)
    deltas = [
        delta_start + (delta_end - delta_start) * i / (n_bins - 1)
        for i in range(n_bins)
    ]

    def get_delta(epoch: int) -> float:
        idx = min(epoch // bin_len, n_bins - 1)
        return deltas[idx]

    return get_delta
