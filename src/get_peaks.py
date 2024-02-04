

"""
get_peaks function finds the peaks in a multimodal distribution
"""


import numpy as np


def make_data(m, n_peaks=3, standerdized=False):
    """
    Makes 1-dimensional multimodal distribution
    """
    means = np.arange(n_peaks) + np.random.normal(scale=0.1, size=n_peaks)
    sigmas = np.random.randint(2, 5, n_peaks) / 10
    pp = [1/n_peaks] * n_peaks
    
    distributions = [np.random.normal(loc=mu, scale=sd, size=round(p*m))
                     for mu, sd, p in zip(means, sigmas, pp)]
    
    a = np.concatenate(distributions)[:m]
    
    if standerdized:
        return (a - a.mean()) / a.std()
    return ((a - a.min() + 1) * 10).round()  # immitates 'house age' in years


def get_peaks(a, n_peaks, return_bins=False):
    """
    Finds n_peaks in distribution 'a'
    sorted by the peaks height
    """
    for n_bins in range(int(len(a)**(1/2)), 5, -1):
        y, x = np.histogram(a, bins=n_bins)
        d = np.sign(y[1:] - y[:-1])
        d2 = d[1:] - d[:-1]
        mask = d2 == -2
        if sum(mask) <= n_peaks:
            break
    
    # Mask to get the peaks
    offset = (len(x)-len(mask)) // 2
    mask = [False]*offset + list(mask) + [False]*(len(x)-len(mask)-offset)
    try:
        idx = np.nonzero(mask)[0]
        peaks = (x[idx] + x[idx+1]) / 2
    except:
        peaks = x[mask]
    
    # Sort the peaks according to height
    heights = y[mask[:len(y)]]
    idx = np.argsort(heights)
    peaks = np.take(peaks, idx)[::-1]
    
    # Return a list or a tuple of lists according to 'return_bins'
    return (peaks, n_bins) if return_bins else peaks



# Demo
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    m = round(np.random.choice(np.logspace(2, 5, num=4, base=10)))
    n_peaks = np.random.randint(1, 7)
    
    a = make_data(m, n_peaks=n_peaks)
    peaks, n_bins = get_peaks(a, n_peaks=n_peaks, return_bins=True)
    print(f"m = {m}\nn_peaks = {n_peaks}\nn_bins = {n_bins}\npeaks: {str(list(peaks.round(1)))[1:-1]}")
    
    plt.hist(a, bins=n_bins)
    _, ymax = plt.ylim()
    plt.vlines(peaks, ymin=0, ymax=ymax, color='k', linewidth=0.9)
    