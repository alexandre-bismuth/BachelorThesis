from torch import arange
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib import cm

##### Start of part adapted from pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")


def plot_spectrogram(specgram, title=None, ylabel="Frequency bins", ax=None):
    # Figure out what the x_axis is since the range is [0,125] - time_bin ? 
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time bins")
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

##### End of part adapted from pytorch documentation

def plot_complex_spectrogram(specgram, title=None, ylabel="freq_bin", plot=None):
    magnitude = np.abs(specgram)
    phase = np.angle(specgram)
    fig, ax = plot if plot is not None else plt.subplots(1, 2)
    if title is not None:
        fig.suptitle(title)
    ax[0].set_ylabel(ylabel)
    ax[0].imshow(librosa.power_to_db(magnitude), origin="lower", aspect="auto", interpolation="nearest")
    ax[0].set_title("Magnitude")
    
    ax[1].imshow(phase, origin="lower", aspect="auto", interpolation="nearest")
    ax[1].set_title("Phase")

# Since this plotting function is used only in specific cases, we reduce its parametrability
def plot_complex_spectrogram_3d(specgram, title=None):
    real_spec, imag_spec = specgram.real, specgram.imag
    freq_bins, time_bins = real_spec.shape
    t, f = np.arange(time_bins), np.arange(freq_bins)
    T_grid, F_grid = np.meshgrid(t, f)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_grid, F_grid, imag_spec, facecolors=plt.cm.plasma(real_spec), rstride=1, cstride=1, linewidth=0)
    mappable = cm.ScalarMappable(cmap=cm.plasma)
    mappable.set_array(real_spec)
    fig = ax.get_figure()
    fig.colorbar(mappable, ax=ax, shrink=0.66, label="Real Part Magnitude")

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("time_bin")
    ax.set_ylabel("freq_bin")
    ax.set_zlabel("Imaginary Part")


    