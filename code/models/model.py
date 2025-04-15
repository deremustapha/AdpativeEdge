import torch
import torch.nn as nn
import torch.nn.functional as F

# from mcunet.mcunet.model_zoo import net_id_list, build_model, download_tflite

class EMGNet(nn.Module):
    def __init__(self, num_gesture):
        super(EMGNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 151) #1280,40
        self.fc2 = nn.Linear(151, num_gesture)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        # x = self.batchnorm1(x)
        x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        return x
    




# def MCUNet():

#     mcunet, _, _ = build_model(net_id="mcunet-in3", pretrained=True)
#     mcunet.first_conv.conv = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     mcunet.classifier = torch.nn.Linear(160, 10)

#     return mcunet
