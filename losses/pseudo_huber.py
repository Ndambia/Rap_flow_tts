def pseudo_huber_distance(x, y, data_dim):
    """d(x,y) = sqrt(||x-y||_2^2 + c^2) - c, with c = 0.00054 * sqrt(d)."""
    c = 0.00054 * (data_dim ** 0.5)
    diff2 = (x - y) ** 2
    diff2 = diff2.flatten(1).sum(-1)               # ||x-y||_2^2 per sample
    return (diff2 + c ** 2).sqrt() - c
