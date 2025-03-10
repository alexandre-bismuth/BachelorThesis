# In this file, we define parent datasets to handle preprocessing

# Machine Learning imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

# Processing imports
import os
import numpy as np

class BaseWaveformDataset(Dataset):
    """
    Parent class for preprocessing. Gets initialized with a dataframe and an augmentation parameters, 
    which determines the probability that each augmentation is applied (or None to bypass)
    """
    def __init__(self, df, augmentation=None):
        self.file_paths = df.index.values
        self.labels = df["label"].values
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform = self.load_audio(idx)
        if self.augmentation is not None:
            waveform = self.augment_wf(waveform, p=self.augmentation)
        label = torch.tensor([self.labels[idx]])
        return waveform, label

    def load_audio(self, idx, normalize=True):
        """Loads and normalizes an audio file."""
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        if normalize: # Time normalization and effect of normalization and eventually remove it for time reasons
            waveform = (waveform - waveform.mean()) / waveform.std()
        return waveform

    def augment_wf(self, waveform, p):
        # Random time shift - prevent detecting gunshots only in the middle of the audio
        if np.random.random() < p: # Check if this actually improves the performance or not - might not improve tbh
            shift_amt = np.random.randint(0, waveform.shape[-1] - 1)
            waveform = torch.roll(waveform, shifts=shift_amt, dims=-1)

        # Gaussian noise - small value since there is already a lot of background noise
        if np.random.random() < p:
            noise_scale = np.random.normal(0, 0.3)
            noise = torch.randn_like(waveform) * noise_scale
            waveform = waveform + noise

        # No need for time stretch, gunshots are sudden
        # No need for pitch shift, gunshots are characteristic + recordings use different rifles
        return waveform


class BaseSpectrogramDataset(BaseWaveformDataset):
    """Parent class for spectrogram-based datasets"""
    def __getitem__(self, idx):
        waveform = self.load_audio(idx)
        if self.augmentation is not None:
            waveform = self.augment_wf(waveform, self.augmentation)
        label = torch.tensor([self.labels[idx]])
        spectrogram = self.process(waveform)
        if self.augmentation is not None:
            spectrogram = self.augment_spec(spectrogram, self.augmentation)
        return spectrogram, label
        
    def augment_spec(self, spectrogram, p):
        # Do Frequency & Time masking, as noise has already been done by augment_wf
        # We choose a lower width for time masking since time shifts have a similar effect
        if np.random.random() < p:
            spectrogram = T.TimeMasking(time_mask_param=10)(spectrogram)

        if np.random.random() < p:
            spectrogram = T.FrequencyMasking(freq_mask_param=20)(spectrogram)
        return spectrogram

    def process(self, waveform):
        raise NotImplementedError("Subclasses of BaseSpectrogramDataset must override the 'process' method.")
