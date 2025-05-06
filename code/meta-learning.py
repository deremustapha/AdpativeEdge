import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
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




def extract_support_n_query(data, labels, n_way, num_support, num_query):

    # N_WAYS = CLASSES
    # NUM_SUPPORT = NUMBER OF SUPPORT DATA PER CLASS
    # NUM_QUERY = NUMBER OF QUERY DATA PER CLASS

    support_query_data = []
    support_query_label = []
    K = np.random.choice(np.unique(labels), n_way, replace=False)
    for i in K:
        index = np.where(labels == i)
        data_per_class = data[index]
        labels_per_class = labels[index[0]]

        permuated_index = np.random.permutation(index[0])
        selected_index = permuated_index[:num_support+num_query]

        support_query_data.append(data[selected_index])
        support_query_label.append(labels[selected_index])
    
    support_query_data = np.array(support_query_data)
    support_query_label = np.array(support_query_label)

    return support_query_data,  support_query_label


def distance(query, support):

    n = query.size(0)
    m = support.size(0)
    d = query.size(1)
    assert d == support.size(1)

    query = query.unsqueeze(1).expand(n, m, d)
    support = support.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(query - support, 2).sum(2)
    return dist


def set_forward_loss(model, data, n_way, n_support, n_query):
    """
    Computes loss, accuracy and output for classification task
    Args:
        sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
    Returns:
        torch.Tensor: shape(2), loss, accuracy and y_hat
    """
    sample_images = data.cuda()
    n_way = n_way
    n_support = n_support
    n_query = n_query

    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]
   
    #target indices are 0 ... n_way-1
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda()
   
    #encode images of the support and the query set
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
   
    z = model(x)
    z_dim = z.size(-1) #usually 64
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
    z_query = z[n_way*n_support:]

    #compute distances
    dists = distance(z_query, z_proto)
    
    #compute probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
   
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
   
    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat
        }


def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
    
    #optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0 #epochs done so far
    stop = False #status to know when to stop
    while epoch < max_epoch and not stop: # Breaks when max epoch is met
        running_loss = 0.0
        running_acc = 0.0
        # For 1 epoch run 2000 episodes
        # 1 Episode is 1 set of n_way classes with n_support and n_query samples each
        #for episode in tnrange(epoch_size, desc="Epoch {:d} train".format(epoch+1)):
        for episode in tqdm(range(epoch_size)):
            sample, _ = extract_support_n_query(train_x, train_y,n_way, n_support, n_query)
            sample = torch.from_numpy(sample).float()
            optimizer.zero_grad()
            model = model.cuda()
            loss, output = set_forward_loss(model, sample, n_way, n_support, n_query)
            running_loss += output['loss']
            running_acc += output['acc']
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print(f'epoch :, {epoch}, loss :, {epoch_loss}, accuracy :, {epoch_acc*100:.2f}%')
        epoch += 1
        scheduler.step()


def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):

    running_loss = 0.0
    running_acc = 0.0

    for episode in tqdm(range(test_episode)):
        sample, _ = extract_support_n_query(test_x, test_y, n_way, n_support, n_query)
        sample = torch.from_numpy(sample).float()
        model = model.cuda()
        loss, output = set_forward_loss(model, sample, n_way, n_support, n_query)
        running_loss += output['loss']
        running_acc += output['acc']
    
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode

    # print(f'Average Loss :, avg_loss, Average Accuracy :, {avg_acc*100:.2f}%')
    return avg_loss, avg_acc


def prepare_meta_data(path, session, subject, stop_subject,
                    num_gesture, num_repetitions,
                    window_time, overlap):
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

    record_time = 5
    fs = 200
    no_channel = 8
    low_cut = 10.0
    high_cut = 99.0
    notch_freq = 60.0
    order = 5
    train_percent=80 
    activate_session = False
    selected_gesture = [1, 2, 3, 4, 5, 6, 7]


    emg_prep = EMGDataPreparation(base_path=path, fs=fs, rec_time=record_time)

   
    train_repetition = np.arange(1, num_repetitions + 1).tolist()
    test_repetition = [1, 2]
  
    train_data, train_labels, _, _ = emg_prep.load_multiple_subject(start_subject=subject, end_subject=stop_subject, session=session, activate_session=activate_session,
            train_repetition=train_repetition, test_repetition=test_repetition, num_gesture=num_gesture, selected_gesture=selected_gesture)


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
    load_path = f"PreTrain_{model_type}_Input_{input_type}_Train_Type_tsts.pth"
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

def run_meta_learning(path, session, start_subject, stop_subject,
                      num_ways, num_support, num_query, input_type, 
                      num_gesture, num_repetitions, window_time, overlap,
                      model_type, max_epoch, epochs, save_path, load_path, seed):

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    X_train, y_train, X_test, y_test = prepare_meta_data(
        path, session, start_subject, stop_subject, 
        num_gesture,  num_repetitions, window_time, overlap)

    # Set random seed
    X_train, X_test = process_input_data(input_type, X_train, X_test)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

     # Create datasets and dataloaders
    
    in_channel = X_train.shape[1]
    set_random_seed(seed)
    model = initialize_model(model_type, input_type, in_channel, num_gesture,
                            device, load_weights=True, weights_path=load_path)



    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training 
    train(model, optimizer, X_train, y_train, num_ways, num_support, num_query, max_epoch, epochs)

    # Testing
    _, test_acc = test(model, X_test, y_test, num_ways, num_support, num_query, int(epochs/2))
    print(f'Meta-Learning Test Accuracy: {test_acc * 100:.2f}%')

        # Save the model
    os.makedirs(save_path, exist_ok=True)
    save_dir = os.path.join(
        save_path,
        #f"MetaLearn_{model_type}_Session_{session}_Subject_{start_subject}_Input_{input_type}_Train_type_{training_type}.pth"
        f"MetaLearn_{model_type}_Input_{input_type}_Train_Type_tsts.pth"
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
    parser.add_argument("--start_subject", type=int, default=1, help="Select Start subject.")
    parser.add_argument("--stop_subject", type=int, default=1, help="Select Stop subject.")
    parser.add_argument("--num_ways", type=int, default=7, help="Number of gestures.")
    parser.add_argument("--num_support", type=int, default=5, help="Number of support samples.")
    parser.add_argument("--num_query", type=int, default=5, help="Number of query samples.")
    parser.add_argument("--input_type", type=str, default='raw', required=True, help="Choose 'raw', 'stft', or 'cwt'.")
    parser.add_argument("--num_gesture", type=int, default=7, help="Number of gestures.")
    parser.add_argument("--num_repetitions", type=int, default=6, help="Number of repetitions.")
    parser.add_argument("--window_time", type=int, default=200, help="Window time in milliseconds.")
    parser.add_argument("--overlap", type=int, default=70, help="Overlap percentage.")
    parser.add_argument("--model_type", type=str, default="EMGNet", help="Model name.")
    parser.add_argument("--max_epoch", type=int, default=15, help="Number of epochs.")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs.")
    parser.add_argument("--save_path", type=str, default="/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights", help="Path to save the model.")
    parser.add_argument("--load_path", type=str, default="/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights", help="Path to save the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_meta_learning(
        args.path, args.session, args.start_subject, args.stop_subject,
        args.num_ways, args.num_support, args.num_query, args.input_type,
        args.num_gesture, args.num_repetitions, args.window_time,  args.overlap, 
        args.model_type, args.max_epoch, args.epochs, args.save_path, args.load_path, args.seed)



if __name__ == "__main__":
    main()