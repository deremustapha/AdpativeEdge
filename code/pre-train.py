import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from sklearn.metrics import (f1_score, recall_score, 
                             precision_score, top_k_accuracy_score)

from utils.data_preparation import *
from utils.preprocessing import *
from models.model import *
from models.mcunet.mcunet.model_zoo import build_model
from utils.optimizers import *
from utils.tools import *
from utils.features import *


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_loop(model, device, data_loader, loss_fn, optimizer):
    """
    Perform one epoch of training.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device to use for training.
        data_loader (DataLoader): DataLoader for training data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.

    Returns:
        tuple: Average training loss and accuracy.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for X, y in data_loader:
        optimizer.zero_grad()
        X, y = X.float().to(device), y.long().to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += y.size(0)
        correct += (y_pred.argmax(1) == y).sum().item()

    return train_loss / total, correct / total


def test_loop(model, device, data_loader, loss_fn):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to use for evaluation.
        data_loader (DataLoader): DataLoader for test data.
        loss_fn (torch.nn.Module): Loss function.

    Returns:
        tuple: Average test loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.float().to(device), y.long().to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_loss += loss.item()
            total += y.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()

    return test_loss / total, correct / total



def prepare_pre_train_data(path, window_time, overlap, num_gesture):

    total_male = 28
    total_female = 12
    no_channels = 8

    fs = 200
    notch_freq=60.0
    low_cut=10.0
    high_cut=99.0 
    order=5
    train_percent = 80 

    X_male, y_male = load_all_cote_participant(path=path, T_participant=total_male, male=True, T_gestures=num_gesture)
    X_female, y_female = load_all_cote_participant(path=path, T_participant=total_female, male=False, T_gestures=num_gesture)


    X = np.concatenate((X_male, X_female), axis=1)
    y = np.concatenate((y_male, y_female), axis=0)
    y = y.astype(int)

    preprocess = EMGPreprocessing(fs=fs, notch_freq=notch_freq, low_cut=low_cut, high_cut=high_cut, order=order)
    X = preprocess.remove_mains(X)
    X = preprocess.highpass_filter(X)

    data, target  = window_with_overlap(data=X, label=y, window_time=window_time, overlap=overlap, no_channel=no_channels, fs=fs)
    data, label = shuffle_data(data=data, labels=target)
    X_train, y_train, X_test, y_test = data_split(data=data, label=label, train_percent=train_percent)
    
    return X_train, y_train, X_test, y_test



def process_input_data(input_type, X_train, X_test):
    """
    Process input data based on the input type.

    Args:
        input_type (str): Input type ('raw', 'stft', or 'cwt').
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Testing data.

    Returns:
        tuple: Processed training and testing data.
    """
    if input_type == "raw":
        X_train = get_raw_data(X_train)
        X_test = get_raw_data(X_test)
    elif input_type == "stft":
        X_train = get_stft_features(X_train)
        X_test = get_stft_features(X_test)
    elif input_type == "cwt":
        X_train = get_cwt_features(X_train)
        X_test = get_cwt_features(X_test)
    else:
        raise ValueError("Invalid input type. Choose 'raw', 'stft', or 'cwt'.")
    return X_train, X_test


def initialize_model(model_type, in_channel, num_gesture, device):
    """
    Initialize the model based on the model type.

    Args:
        model_type (str): Model type ('EMGNet', 'EMGNas', or 'MCUNet').
        in_channel (int): Number of input channels.
        num_gesture (int): Number of gestures.
        device (torch.device): Device to use.

    Returns:
        torch.nn.Module: Initialized model.
    """
    if model_type == "EMGNet":
        return EMGNet(in_channel, num_gesture).to(device)
    elif model_type == "EMGNetFAN":
        return EMGNetFAN(in_channel, num_gesture).to(device)
    elif model_type == "EMGNas":
        return EMGNas(in_channel, num_gesture).to(device)
    elif model_type == "EMGNasFAN":
        return EMGNasFAN(in_channel, num_gesture).to(device)
    elif model_type == "MCUNet":
        def MCUNet(input_channel, number_gestures):
            mcunet, _, _ = build_model(net_id="mcunet-in3", pretrained=True)
            mcunet.first_conv.conv = torch.nn.Conv2d(input_channel, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            mcunet.classifier = torch.nn.Linear(160, number_gestures)
            return mcunet
        return MCUNet(in_channel, num_gesture).to(device)
    elif model_type == "ProxyLessNas":
        return ProxyLessNas(in_channel, num_gesture).to(device)
    elif model_type == "MobileNet":
        return MobileNet(in_channel, num_gesture).to(device)
    else:
        raise ValueError("Invalid model type. Choose 'EMGNet', 'EMGNas', or 'MCUNet'.")


def run_pretrain(path, input_type, num_gesture, window_time, overlap, model_type, epochs, save_path, seed):
    """
    Run the pretraining process.

    Args:
        path (str): Path to the dataset.
        session (int): Session number.
        subject (int): Subject number.
        num_repetitions (int): Number of repetitions.
        input_type (str): Input type ('raw', 'stft', or 'cwt').
        num_gesture (int): Number of gestures.
        model_type (str): Model type ('EMGNet', 'EMGNas', or 'MCUNet').
        epochs (int): Number of epochs.
        save_path (str): Path to save the model.
        seed (int): Random seed.
    """
    # Hyperparameters

    batch_size = 128
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    X_train, y_train, X_test, y_test = prepare_pre_train_data(path, window_time, overlap, num_gesture)


    # Data preprocessing
    X_train, X_test = process_input_data(input_type, X_train, X_test)

    print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

    # Create datasets and dataloaders
    train_dataset = EMGDataset(X_train, y_train)
    test_dataset = EMGDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model initialization
    in_channel = X_train.shape[1]
    set_random_seed(seed)
    model = initialize_model(model_type, in_channel, num_gesture, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        _, train_acc = train_loop(model, device, train_dataloader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {train_acc * 100:.2f}%")

    # Testing loop
    _, test_acc = test_loop(model, device, test_dataloader, criterion)
    print(f"The PreTraining Test Accuracy is: {test_acc * 100:.2f}%")

    # Save the model
    os.makedirs(save_path, exist_ok=True)
    save_dir = os.path.join(
        save_path,
        f"PreTrain_{model_type}_Input_{input_type}.pth"
    )
    torch.save(model.state_dict(), save_dir)
    print(f"Model saved to {save_dir}")


def main():
    """
    Main function to parse arguments and run the pretraining process.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate the EMGNet model.")
    parser.add_argument("--path", type=str, default='/mnt/d/AI-Workspace/sEMGClassification/AdaptiveModel/data/6_Flex_BMIS/flex_bmis/mat_data', required=True, help="Path to the dataset.")
    parser.add_argument("--input_type", type=str, default='raw', required=True, help="Choose 'raw', 'stft', or 'cwt'.")
    parser.add_argument("--num_gesture", type=int, default=7, help="Number of gestures.")
    parser.add_argument("--window_time", type=int, default=200, help="Number of repetitions.")
    parser.add_argument("--overlap", type=int, default=80, help="Number of repetitions.")
    parser.add_argument("--model_type", type=str, default="EMGNet", help="Model name.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--save_path", type=str, default="/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights", help="Path to save the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_pretrain(
        args.path, args.input_type,args.num_gesture, args.window_time, args.overlap,
        args.model_type, args.epochs, args.save_path, args.seed
    )


if __name__ == "__main__":
    main()

# python3 pre-train.py \
#     --path '/mnt/d/AI-Workspace/sEMGClassification/AdaptiveModel/data/1_MyoArmbandDataset/PreTrain' \
#     --input_type raw \
#     --training_type tsts \
#     --session 1 \
#     --subject 1 \
#     --num_repetitions 9 \
#     --num_gesture 7 \
#     --model_type EMGNet \
#     --epochs 50 \
#     --save_path '/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights/PreTrain' \
#     --seed 42