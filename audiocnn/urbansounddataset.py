import os
from torch.utils.data import Dataset
import torch
import pandas as pd
import torchaudio
############## Parameters ##################
ANNOTATIONS_FILE = "/Users/vilourenco/datasets/UrbanSound8K/metadata/UrbanSound8K.csv" # Path to the csv file containing the annotations
AUDIO_DIR = "/Users/vilourenco/datasets/UrbanSound8K/audio" # Path to the directory containing the audio samples
SAMPLE_RATE = 22050 # The sampling rate of the audio samples
NUM_SAMPLES = 22050 # The number of samples to retain in the audio samples
###############  Dataset  ##################
class UrbanSoundDataset(Dataset):

    # __init__ is the constructor of the class
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        """
            annotations_file: path to the csv file containing the annotations
            audio_dir: path to the directory containing the audio samples
        """
        # Read the csv file with the annotations
        # Get the full path to the audio files
        # Get the device
        # Get the transformation, and send it to the correct device
        # Get the target sample rate
        # Get the number of samples
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        

    def __len__(self):
        """
            Returns:
                The length of the dataset
        """
        # Return the length of the annotations dataframe
        return len(self.annotations)

    def __getitem__(self, index):
        """
            index: the index of the item to return
            Returns:
                The audio sample and its corresponding label
        """
        # Get the name of the audio sample from the pandas df
        # Load the audio sample
        # Get the audio signal tensor and the sampling frequency
        # Send the signal to the right device (for GPU processing)
        # Resample the signal if necessary
        # If the signal has multiple channels, lower to one channel
        # Cut the signal to the desired length
        # Pad the signal to the desired length
        # Use a transformation over the audio signal (e.g., MelSpectrogram)
        audio_sample_path = self.__get_audio_sample_path(index)
        label = self.__get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)   # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = self.__resample_if_necessary(signal, sr)
        signal = self.__mix_down_if_necessary(signal)
        signal = self.__cut_if_necessary(signal)
        signal = self.__right_pad_if_necessary(signal)
        signal = self.transformation(signal)


        return signal, label
    
        # Private method to mix down the signal if necessary
    def __resample_if_necessary(self, signal, sr):
        """
            signal: the audio signal tensor
            sr: the sampling rate of the audio signal
            Returns:
                The audio signal tensor resampled at the target sampling rate
        """
        # If the sampling rate of the signal and the target sampling rate are different, resample
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler = resampler
            signal = resampler(signal)
        return signal
    
    # Private method to mix down the signal if necessary
    def __mix_down_if_necessary(self, signal):
        """
            signal: the audio signal tensor
            sr: the sampling rate of the audio signal
            Returns:
                The audio signal tensor with one channel
        """
        # If the signal has more than one channel, mix it down
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def __cut_if_necessary(self, signal):
        """
            signal: the audio signal tensor
            Returns:
                The audio signal tensor with the desired number of samples
        """
        # signal -> Tensor -> (1, num_samples)

        # If the signal has more samples than the desired number of samples, cut it
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def __right_pad_if_necessary(self, signal):
        """
            signal: the audio signal tensor
            Returns:
                The audio signal tensor padded with zeros to the desired number of samples
        """
        # signal -> Tensor -> (1, num_samples)

        # If the signal has less samples than the desired number of samples, pad it
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # Calculate the number of missing samples
            # Create the padding, (left_pad, right_pad, top_pad, bottom_pad)
            # Pad the signal on the left with zeros
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def __get_audio_sample_path(self, index):
        """
        Private Method
            index: the index of the item to return
            Returns:
                The path to the audio file
        """
        # Get the fold number from the pandas df
        fold = f"fold{self.annotations.iloc[index, 5]}"
        # Get the name of the audio file
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def __get_audio_sample_label(self, index):
        """
        Private Method
            index: the index of the item to return
            Returns:
                The label of the audio file
        """
        # Get the label from the pandas df
        return self.annotations.iloc[index, 6]


def check_device():
    """
        Check if Metal / GPU is available and set the device accordingly
        Returns:
            The device to use for training
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available(): 
        device = "cuda"
    else: 
        device = "cpu"
    
    return device

###############  Main  ##################
if __name__ == "__main__":

    #Check device
    device = check_device()
    print(f"Using {device} device. PS: Not using during dataset preprocessing due to incompatibility")

    # Create the mel spectrogram transform
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # Create the dataset
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device)

    # Print the length of the dataset
    print(f"There are {len(usd)} samples in the dataset.")

    # Get the first sample and its corresponding label
    signal, label = usd[1]
    print(f"Shape of the signal: {signal.shape}")