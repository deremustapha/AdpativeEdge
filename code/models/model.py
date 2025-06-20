import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights



class FANLayer(nn.Module):

    def __init__(self, input_dim, output_dim, p_ratio=0.25):
        super(FANLayer, self).__init__()
        
        # Ensure the p_ratio is within a valid range
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms

        # Linear transformation for the p component (for cosine and sine parts)
        #self.input_linear_p = BinarizeLinear(input_dim, p_output_dim)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim)
        
        # Linear transformation for the g component
        #self.input_linear_g = BinarizeLinear(input_dim, g_output_dim)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        #self.activation = nn.Hardtanh()
        self.activation = nn.GELU()
        

    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), after applying the FAN layer.
        """
        
        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(src))
        # Apply the linear transformation for the p component
        p = self.input_linear_p(src)
        # Concatenate cos(p), sin(p), and activated g along the last dimension
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)

        
        return output
    
def binarize(tensor):
    return tensor.sign()

class BinarizeLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(BinarizeLinear, self).__init__(in_features, out_features)

    def forward(self, input):
        # input * weight
        
        # binarize input
        input.data = binarize(input.data) # Binarize the tensor

        # binarize weight
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
            
        self.weight.data = binarize(self.weight.org)

        res = nn.functional.linear(input, self.weight)

        return res


class BinarizeConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(BinarizeConv, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

    def forward(self, input):
        # input * weight
        
        # binarize input
        input.data = binarize(input.data) # Binarize the tensor

        # binarize weight
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
            
        self.weight.data = binarize(self.weight.org)

        res = nn.functional.conv2d(input, self.weight)

        return res

# Define the EMGNet model
class EMGNet(nn.Module):
    def __init__(self, in_channel, num_gesture):
        super(EMGNet, self).__init__()
        self.initial = nn.Conv2d(in_channel, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 516) # raw 1152 # stft 512
        self.relu4 = nn.ReLU()
        self.last = nn.Linear(516, num_gesture)

    def forward(self, x):
        x = self.initial(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.last(x)
        return x


# Define the 1 bit Quantized EMGNet model
class EMGNetQuantized(nn.Module):
    def __init__(self, in_channel, num_gesture):
        super(EMGNetQuantized, self).__init__()
        self.initial = nn.Conv2d(in_channel, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.Hardtanh()
        self.conv2 = BinarizeConv(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.Hardtanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = BinarizeLinear(1152, 516) # raw 1152 # stft 512
        self.bn3 = nn.BatchNorm2d(516)
        self.relu3 = nn.Hardtanh()
        self.last = nn.Linear(516, num_gesture)

    def forward(self, x):
        x = self.initial(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.last(x)
        return x


class EMGNetFAN(nn.Module):
    def __init__(self, in_channel, num_gesture):
        super(EMGNetFAN, self).__init__()
        similarparameter=False
        self.similarparameter = similarparameter
        self.initial = nn.Conv2d(in_channel, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.scalar = lambda x: x*4//3 if self.similarparameter else x
        self.fc1 = FANLayer(1152, self.scalar(256))  # raw 1152 # stft 512
        self.last = nn.Linear(256, num_gesture)

    def forward(self, x):
        x = self.initial(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.last(x)
        return x


class EMGNetFAN(nn.Module):
    def __init__(self, in_channel, num_gesture):
        super(EMGNetFAN, self).__init__()
        similarparameter=False
        self.similarparameter = similarparameter
        self.initial = nn.Conv2d(in_channel, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.scalar = lambda x: x*4//3 if self.similarparameter else x
        self.fc1 = FANLayer(1152, self.scalar(256))  # raw 1152 # stft 512
        self.last = nn.Linear(256, num_gesture)

    def forward(self, x):
        x = self.initial(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.last(x)
        return x
    


class EMGNas(nn.Module):
    def __init__(self, in_channel, num_gesture):
        super(EMGNas, self).__init__()
        similarparameter=False
        self.similarparameter = similarparameter

        self.num_gesture = num_gesture

        self.initial = nn.Conv2d(in_channel, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(6, 9, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9, 13) #1280,40
        self.relu3 = nn.ReLU()
        self.last = nn.Linear(13, self.num_gesture)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.last(x)
        return x
    


class EMGNasQuantized(nn.Module):
    def __init__(self, in_channel, num_gesture):
        super(EMGNasQuantized, self).__init__()
        similarparameter=False
        self.similarparameter = similarparameter

        self.num_gesture = num_gesture

        self.initial = nn.Conv2d(in_channel, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.Hardtanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = BinarizeConv(6, 9, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.Hardtanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = BinarizeLinear(9, 13)
        self.relu3 = nn.Hardtanh()
        self.last = nn.Linear(13, self.num_gesture)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.last(x)
        return x
    

class EMGNasFAN(nn.Module):
    def __init__(self, in_channel, num_gesture):
        super(EMGNasFAN, self).__init__()
        similarparameter=False
        self.similarparameter = similarparameter
        self.num_gesture = num_gesture

        self.initial = nn.Conv2d(in_channel, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(6, 9, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.scalar = lambda x: x*4//3 if self.similarparameter else x
        self.fc1 = FANLayer(9, self.scalar(256))  # raw =180, stft=90
        self.last = nn.Linear(256, self.num_gesture)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.last(x)
        return x


class EMGNasFANQuantized(nn.Module):
    def __init__(self, in_channel, num_gesture):
        super(EMGNasFANQuantized, self).__init__()
        similarparameter=False
        self.similarparameter = similarparameter
        self.num_gesture = num_gesture

        self.initial = nn.Conv2d(in_channel, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.Hardtanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(6, 9, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.Hardtanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.scalar = lambda x: x*4//3 if self.similarparameter else x
        self.fc1 = FANLayer(9, self.scalar(256))  # raw =180, stft=90
        self.last = nn.Linear(256, self.num_gesture)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.last(x)
        return x


def MobileNet(input_channel, number_gestures):
    
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights)
    model.features[0][0] = nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=number_gestures, bias=True)
    return model


def ProxyLessNas(input_channel, number_gestures):
    target_platform = "proxyless_cpu"
    model = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=True)
    model.first_conv.conv = nn.Conv2d(input_channel, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier = nn.Linear(1432, number_gestures)

    return model


class CTRLEMGNet(nn.Module):
    def __init__(self, num_gesture):
        super(CTRLEMGNet, self).__init__()

        self.num_gesture = num_gesture

        self.conv1 = nn.Conv2d(1, 32, kernel_size=1, stride=2)
        self.lstm1 = nn.LSTM(32, 32, 2, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, 2, batch_first=True)
        self.lstm3 = nn.LSTM(32, 32, 2, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 32)
        self.output = nn.Linear(32, self.num_gesture)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.lstm1(x)
        print(x.shape)
        x = self.lstm2(x)
        print(x.shape)
        x = self.lstm3(x)
        x = self.flatten(x)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
    
class CTRLEMG(nn.Module):
    def __init__(self, input_channels, conv_filters, lstm_hidden_size, lstm_layers, num_classes, stride=10):
        super(CTRLEMG, self).__init__()

        self.conv1d = nn.Conv1d(
            in_channels=input_channels,  # This should match the channel dimension of your input
            out_channels=conv_filters,
            kernel_size=10,
            stride=stride
        )
        self.layer_norm1 = nn.LayerNorm(conv_filters)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(lstm_hidden_size)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # Remove the channel dimension (from shape [batch_size, 1, 128, 2256] to [batch_size, 128, 2256])
        # Now the input tensor has the shape [batch_size, 128, 2256]
        x = self.conv1d(x)  # Apply Conv1D
        x = self.layer_norm1(x.permute(0, 2, 1))  # Layer normalization
        x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.layer_norm2(x)
        x = x[:, -1, :]  # Get the output from the last timestep
        x = self.fc(x)
        return x
    

# Example usage:
# model = GestureRecognitionModel(input_dim=128, num_gestures=9)
# input_tensor = torch.randn(32, 128)  # Batch size of 32, input dimension of 128
# output = model(input_tensor)
# python fine-tune_1_bit.py --path "/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/data/6_Flex_BMIS/flex_bmis/mat_data" --session 1 --subject 7 --input "raw" --num_gesture 7 --num_repetitions 9 --window_time 160 --overlap 70 --training_type tsts --model_type ProxyLessNas --epochs 30 --save_path "/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights/FineTune/WithPre_Train" --load_path "/mnt/d/AI-Workspace/sEMGClassification/EdgeLastTrain/model_weights/PreTrain"