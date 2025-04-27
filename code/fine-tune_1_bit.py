from utils.data_preparation import *
from utils.preprocessing import *
from models.model import *
from models.mcunet.mcunet.model_zoo import build_model
from utils.optimizers import *
from utils.tools import *
from utils.features import *


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



def fine_tune_loop(model, train_device, data, loss_fn, optimizer):

    model.train()
    train_loss = 0
    correct = 0
    total = 0


    for X, y in data:


            optimizer.zero_grad()

            X = X.float().to(train_device)
            y = y.long().to(train_device)
            model = model.to(train_device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            #scheduler.step()

            train_loss += loss.item()
            total += y.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()


    return train_loss / total, correct / total


def test_loop(model, train_device, data, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in data:
            X = X.float().to(train_device)
            y = y.long().to(train_device)
            model = model.to(train_device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_loss += loss.item()
            total += y.size(0)
            correct += (y_pred.argmax(1) == y).sum().item()

    return test_loss / total, correct / total




def LRO_train_test_split(num_repetitions):
    """
    Perform Leave-Repetition-Out (LRO) train-test split.

    Args:
        num_repetitions (int): Total number of repetitions.

    Returns:
        tuple: Train and test repetition indices.
    """
    num_rep = np.arange(1, num_repetitions + 1).tolist()
    test_numbers = random.sample(num_rep, k=int(len(num_rep) * 0.3))
    train_numbers = [n for n in num_rep if n not in test_numbers]
    return train_numbers, test_numbers

def prepare_data(path, session, subject, num_gesture, num_repetitions,
                training_type, window_time, overlap):
    """
    Prepare training and testing data based on the training type.

    Args:
        path (str): Path to the dataset.
        session (int): Session number.
        subject (int): Subject number.
        num_repetitions (int): Number of repetitions.
        training_type (str): Training type ('TSTS' or 'LRO').
        num_gesture (int): Number of gestures.
        selected_gesture (list): List of selected gestures.
        record_time (int): Recording time in seconds.
        fs (int): Sampling frequency.
        notch_freq (float): Notch filter frequency.
        low_cut (float): Low cutoff frequency.
        high_cut (float): High cutoff frequency.
        order (int): Filter order.
        window_time (int): Window time in milliseconds.
        overlap (int): Overlap percentage.
        no_channel (int): Number of channels.
        activate_session (bool): Whether to activate session.

    Returns:
        tuple: Training and testing data and labels.
    """
    record_time=5
    fs=250
    no_channel=8
    low_cut = 10.0
    high_cut=120.0
    notch_freq=60.0
    order=5
    train_percent=80 
    activate_session = True
    selected_gesture = [1, 2, 3, 4, 5, 6, 7]

    emg_prep = EMGDataPreparation(base_path=path, fs=fs, rec_time=record_time)

    if training_type == "tsts":
        train_repetition = np.arange(1, num_repetitions + 1).tolist()
        test_repetition = [1, 2]
        subject_path, train_gesture, test_gesture = emg_prep.get_per_subject_file(
            subject_number=subject, num_gesture=num_gesture, session=session, activate_session=activate_session,
            train_repetition=train_repetition, test_repetition=test_repetition
        )
        train_data, _ = emg_prep.load_data_per_subject(
            subject_path, selected_gesture=selected_gesture, train_gesture=train_gesture, test_gesture=test_gesture
        )
        train_data, train_labels = emg_prep.get_data_labels(train_data)
        preprocess = EMGPreprocessing(fs=fs, notch_freq=notch_freq, low_cut=low_cut, high_cut=high_cut, order=order)
        train_data = preprocess.remove_mains(train_data)
        train_data = preprocess.highpass_filter(train_data)
        window_train_data, window_train_labels = emg_prep.window_with_overlap(
            train_data, train_labels, window_time=window_time, overlap=overlap, no_channel=no_channel)
        window_train_data, window_train_labels = shuffle_data(window_train_data, window_train_labels)   
        window_train_data, window_train_labels, window_test_data, window_test_labels = data_split(
                window_train_data, window_train_labels, train_percent=train_percent) 
        X_train, y_train = shuffle_data(window_train_data, window_train_labels)
        X_test, y_test = shuffle_data(window_test_data, window_test_labels) 
        return X_train, y_train, X_test, y_test 
    
    elif training_type == "lro":
        train_repetition, test_repetition = LRO_train_test_split(num_repetitions)
        subject_path, train_gesture, test_gesture = emg_prep.get_per_subject_file(
            subject_number=subject, num_gesture=num_gesture, session=session, activate_session=activate_session,
            train_repetition=train_repetition, test_repetition=test_repetition
        )
        train_data, test_data = emg_prep.load_data_per_subject(
            subject_path, selected_gesture=selected_gesture, train_gesture=train_gesture, test_gesture=test_gesture
        )
        train_data, train_labels = emg_prep.get_data_labels(train_data)
        test_data, test_labels = emg_prep.get_data_labels(test_data)

        preprocess = EMGPreprocessing(fs=fs, notch_freq=notch_freq, low_cut=low_cut, high_cut=high_cut, order=order)
        train_data = preprocess.remove_mains(train_data)
        test_data = preprocess.remove_mains(test_data)
        train_data = preprocess.highpass_filter(train_data)
        test_data = preprocess.highpass_filter(test_data)

        window_train_data, window_train_labels = emg_prep.window_with_overlap(
            train_data, train_labels, window_time=window_time, overlap=overlap, no_channel=no_channel
        )
        window_test_data, window_test_labels = emg_prep.window_with_overlap(
            test_data, test_labels, window_time=window_time, overlap=overlap, no_channel=no_channel
        )

        X_train, y_train = shuffle_data(window_train_data, window_train_labels)
        X_test, y_test = shuffle_data(window_test_data, window_test_labels)

        return X_train, y_train, X_test, y_test
    else:
        raise ValueError("Invalid training type. Choose 'TSTS' or 'LRO'.")


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


def initialize_model(model_type, input_type, training_type, 
                    in_channel, num_gesture, device, load_weights,
                    weights_path):
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
    load_path = f"KD_{model_type}_Input_{input_type}_Train_Type_{training_type}.pth"
    weights_path = os.path.join(weights_path, load_path)
    if model_type == "EMGNet":
        if load_weights:
            model = EMGNet(in_channel, num_gesture).to(device)
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
            return model
        else:
            print("Loading default weights")
            return EMGNet(in_channel, num_gesture).to(device)
    elif model_type == "EMGNetFAN":
        if load_weights:
            model = EMGNetFAN(in_channel, num_gesture).to(device)
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
            return model
        else:
            print("Loading default weights")
            return EMGNetFAN(in_channel, num_gesture).to(device)
    elif model_type == "EMGNas":
        if load_weights:
            model = EMGNas(in_channel, num_gesture).to(device)
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
            return model
        else:
            print("Loading default weights")
            return EMGNas(in_channel, num_gesture).to(device)
    elif model_type == "EMGNasFAN":
        if load_weights:
            model = EMGNasFAN(in_channel, num_gesture).to(device)
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
            return model
        else:
            print("Loading default weights")
            return EMGNasFAN(in_channel, num_gesture).to(device)
    elif model_type == "MCUNet":
        def MCUNet(input_channel, number_gestures):
            mcunet, _, _ = build_model(net_id="mcunet-in3", pretrained=True)
            mcunet.first_conv.conv = torch.nn.Conv2d(input_channel, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            mcunet.classifier = torch.nn.Linear(160, number_gestures)
            return mcunet
        if load_weights:
            model = MCUNet(in_channel, num_gesture).to(device)
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
            return model
        else:
            print("Loading default weights")
            return MCUNet(in_channel, num_gesture).to(device)
    elif model_type =="MobileNet":
        if load_weights:
            model = MobileNet(in_channel, num_gesture).to(device)
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
            return model
        else:
            print("Loading default weights")
            return MobileNet(in_channel, num_gesture).to(device)
    elif model_type == "ProxyLessNas":
        if load_weights:
            model = ProxyLessNas(in_channel, num_gesture).to(device)
            model.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
            return model
        else:
            print("Loading default weights")
            return ProxyLessNas(in_channel, num_gesture).to(device)
    else:
        raise ValueError("Invalid model type. Choose 'EMGNet', 'EMGNas', or 'MCUNet'.")
    



def run_fine_tune(path, session, subject, input_type, num_gesture,
                num_repetitions, window_time, overlap, training_type, 
                model_type, epochs, save_path, load_path, seed):
    


    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_random_seed(seed)

       # Data preparation
    X_train, y_train, X_test, y_test = prepare_data(path, session, subject,
                                                    num_gesture, num_repetitions,
                                                    training_type,  window_time,
                                                    overlap)

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

    ####### None #########################################################
    model = initialize_model(model_type, input_type, training_type,in_channel,
                            num_gesture, device, load_weights=True, weights_path=load_path)

    print(f"Without Fine-Tunning")
    _, test_acc = test_loop(model, device, test_dataloader, criterion)
    print(f'Test accuracy {test_acc*100:.4f}%')
    ################################################################################

    ####### Full-Train #########################################################
    model = initialize_model(model_type, input_type, training_type,in_channel,
                            num_gesture, device, load_weights=True, weights_path=load_path)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = fine_tune_loop(model, device, train_dataloader, criterion, optimizer)
    
    _, test_acc = test_loop(model, device, test_dataloader, criterion)
    print(f"Full Layer Fine-Tunning")
    print(f'Test Accuracy Full-Train: {test_acc*100:.4f}%')
    ################################################################################

    ####### TinyTL ######
    model = initialize_model(model_type, input_type, training_type,in_channel,
                            num_gesture, device, load_weights=True, weights_path=load_path)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for name, param in model.named_parameters():
    
        if 'weight' in name:
            param.requires_grad = False # Freeze the weights
        elif 'bias' in name:
            param.requires_grad = True # Unfreeze the bias
    
    for epoch in tqdm(range(epochs)):
    
        _, train_acc = fine_tune_loop(model, device, train_dataloader, criterion, optimizer)

    _, test_acc = test_loop(model, device, test_dataloader, criterion)
    print(f"TinyTL  Train Fine-Tunning")
    print(f'Test Accuracy TinyTL: {test_acc*100:.4f}%')
    
    ################################################################################

    ####### Adaptive Edge Update ######
    model = initialize_model(model_type, input_type, training_type,in_channel,
                            num_gesture, device, load_weights=True, weights_path=load_path)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    last_layer = str(list(model.named_children())[-1][0])
    print(f"Last Layer: {last_layer}")

    for name, param in model.named_parameters():
        # print("Name: ", name)
        if name.startswith(last_layer):
            # print(f"Unfreezing {last_layer} Layer")
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    for epoch in tqdm(range(epochs)):
    
        _, train_acc = fine_tune_loop(model, device, train_dataloader, criterion, optimizer)

    _, test_acc = test_loop(model, device, test_dataloader, criterion)
    print(f"Adaptive Edge Last  Train Fine-Tunning")
    print(f'Test Accuracy Last-Train: {test_acc*100:.4f}%')
    ################################################################################

    # Save the student model
    os.makedirs(save_path, exist_ok=True)
    save_dir = os.path.join(
        save_path,
        f"FineTuned_{model_type}_Input_{input_type}_Train_Type_{training_type}.pth"
    )

    torch.save(model.state_dict(), save_dir)
    print(f"Model saved to {save_dir}")


def main():
    """
    Main function to parse arguments and run the pretraining process.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate the EMGNet model.")
    parser.add_argument("--path", type=str, default='/mnt/d/AI-Workspace/sEMGClassification/AdaptiveModel/data/6_Flex_BMIS/flex_bmis/mat_data', required=True, help="Path to the dataset.")
    parser.add_argument("--session", type=int, default=1, help="Select one of four sessions.")
    parser.add_argument("--subject", type=int, default=3, help="Select subject.")
    parser.add_argument("--input_type", type=str, default='raw', required=True, help="Choose 'raw', 'stft', or 'cwt'.")
    parser.add_argument("--num_gesture", type=int, default=7, help="Number of gestures.")
    parser.add_argument("--num_repetitions", type=int, default=8, help="Number of repetitions.")   
    parser.add_argument("--window_time", type=int, default=160, help="Window time in milliseconds.")
    parser.add_argument("--overlap", type=int, default=70, help="Overlap percentage.")  
    parser.add_argument("--training_type", type=str, default='tsts', required=True, help="Choose 'TSTS' or 'LRO'.")
    parser.add_argument("--model_type", type=str, default="EMGNasFAN", help="Model name.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--save_path", type=str, default="/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights/FineTune", help="Path to save the model.")
    parser.add_argument("--load_path", type=str, default="/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights/KD", help="Path to save the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()


    run_fine_tune(args.path, args.session, args.subject, args.input_type,
                    args.num_gesture, args.num_repetitions, args.window_time,
                    args.overlap, args.training_type, args.model_type, args.epochs, 
                    args.save_path, args.load_path, args.seed)


if __name__ == "__main__":
    main()


