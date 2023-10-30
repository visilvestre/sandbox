import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/Users/vilourenco/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "/Users/vilourenco/datasets/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    """
        train_data: the dataset to load
        batch_size: the size of the batch
        Returns:
            A DataLoader that can be used in a for loop
    """
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    """
        model: the model to train
        data_loader: the DataLoader object to iterate over the dataset 
        loss_fn: the loss function to use
        optimiser: the optimiser to use to update the model weights
        device: the device to use (e.g., cpu, cuda)
    """
    for input, target in data_loader:
        #input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    """
        model: the model to train
        data_loader: the DataLoader object to iterate over the dataset
        loss_fn: the loss function to use
        optimiser: the optimiser to use to update the model weights
        device: the device to use (e.g., cpu, cuda)
        epochs: the number of epochs to train the model
    """
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

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

if __name__ == "__main__":
    
    # Check device to use
    #device = check_device()
    device = "cpu"
    print(f"Using {device}")

    # Instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device="cpu" if device == "mps" else device) #Change device to cpu if using MPS
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # Construct model and assign it to device
    cnn = CNNNetwork() #.to(device)
    print(cnn)

    # Initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")