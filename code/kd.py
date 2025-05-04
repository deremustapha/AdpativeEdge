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

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate)
    
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student.train() # Student to train mode
            student_logits = student(inputs)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.float().to(device), labels.long().to(device)

                outputs = student(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total


        print(f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.2f}%")
    

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

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


def initialize_model(model_type, input_type, in_channel, num_gesture, 
                     device, load_weights, weights_path):
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
    load_path = f"MetaLearn_{model_type}_Input_{input_type}.pth"
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


def run_kd(path, session, subject, input_type, num_gesture, 
           num_repetitions, window_time, overlap, training_type, 
            model_type, epochs, save_path, load_path, seed):
    
    """
    Run knowledge distillation.

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
        input_type (str): Input type ('raw', 'stft', or 'cwt').
        model_type (str): Model type ('EMGNet', 'EMGNas', or 'MCUNet').
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        T (float): Temperature for knowledge distillation.
        soft_target_loss_weight (float): Weight for the soft target loss.
        ce_loss_weight (float): Weight for the cross-entropy loss.

    Returns:
        None
    """

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    T=3
    soft_target_loss_weight=0.25
    ce_loss_weight=0.75
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # Set random seed
    set_random_seed(seed)

# prepare_data(path, session, subject, num_gesture, num_repetitions,
#                 training_type, window_time, overlap)

   # Data preparation
    X_train, y_train, X_test, y_test = prepare_data(
        path, session, subject, num_gesture, num_repetitions,
        training_type,  window_time, overlap)

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
    teacher_model_type = 'MCUNet'
    teacher_model = initialize_model(teacher_model_type, input_type, in_channel,
                                    num_gesture, device, load_weights=True, 
                                    weights_path=load_path)
    print(f"Teacher Model Loaded")

    set_random_seed(seed)
    student_model = initialize_model(model_type, input_type, in_channel,
                                    num_gesture, device, load_weights=True, 
                                    weights_path=load_path)
    print(f"Student Model Loaded")



    train_knowledge_distillation(teacher=teacher_model, student=student_model, 
                                 train_loader=train_dataloader, epochs=epochs, 
                                 learning_rate=learning_rate, T=T, 
                                 soft_target_loss_weight=soft_target_loss_weight, 
                                 ce_loss_weight=ce_loss_weight, device=device)
    
    # Save the student model
    os.makedirs(save_path, exist_ok=True)
    save_dir = os.path.join(
        save_path,
        f"KD_{model_type}_Input_{input_type}_Train_Type_{training_type}.pth"
    )
    torch.save(student_model.state_dict(), save_dir)
    print(f"Model saved to {save_dir}")

    test_accuracy = test(student_model, test_dataloader, device)
    print(f"Knowledge Distillation Test Accuracy: {test_accuracy:.2f}%")


def main():
    """
    Main function to parse arguments and run the pretraining process.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate the EMGNet model.")
    parser.add_argument("--path", type=str, default='/mnt/d/AI-Workspace/sEMGClassification/AdaptiveModel/data/6_Flex_BMIS/flex_bmis/mat_data', required=True, help="Path to the dataset.")
    parser.add_argument("--session", type=int, default=1, help="Select one of four sessions.")
    parser.add_argument("--subject", type=int, default=1, help="Select subject.")
    parser.add_argument("--input_type", type=str, default='raw', required=True, help="Choose 'raw', 'stft', or 'cwt'.")
    parser.add_argument("--num_gesture", type=int, default=7, help="Number of gestures.")
    parser.add_argument("--num_repetitions", type=int, default=9, help="Number of repetitions.")
    parser.add_argument("--window_time", type=int, default=200, help="Window time in milliseconds.")
    parser.add_argument("--overlap", type=int, default=70, help="Overlap percentage.")
    parser.add_argument("--training_type", type=str, default='tsts', required=True, help="Choose 'TSTS' or 'LRO'.")
    parser.add_argument("--model_type", type=str, default="EMGNet", help="Model name.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--save_path", type=str, default="/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights", help="Path to save the model.")
    parser.add_argument("--load_path", type=str, default="/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights", help="Path to save the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()


    run_kd(
        args.path, args.session, args.subject, args.input_type,
        args.num_gesture, args.num_repetitions, args.window_time,
        args.overlap, args.training_type, args.model_type, args.epochs, 
        args.save_path, args.load_path, args.seed)


if __name__ == "__main__":
    main()