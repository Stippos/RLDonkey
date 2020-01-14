import torch.nn as nn
import torch.nn.functional as F

class EncoderDeepConv(nn.Module):
    def __init__(self, n_components):
        super(EncoderDeepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(14*14*16, 250)
        self.fc2 = nn.Linear(250, n_components)

    def forward(self, x, verbose=False):
        if verbose: print('Encoder')
        if verbose: print(x.shape)
        x = F.relu(self.conv1(x))
        if verbose: print(x.shape)
        x = F.relu(self.conv2(x))
        if verbose: print(x.shape)
        x = x.view(-1, self.fc1.in_features)
        if verbose: print(x.shape)
        x = F.relu(self.fc1(x))
        if verbose: print(x.shape)
        x = self.fc2(x)
        if verbose: print(x.shape)
        return x


class DecoderDeepConv(nn.Module):
    def __init__(self, n_components):
        super(DecoderDeepConv, self).__init__()        
        self.fc1 = nn.Linear(n_components, 250)
        self.fc2 = nn.Linear(250, 14*14*16)
        self.conv1 = nn.ConvTranspose2d(in_channels = 16, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.ConvTranspose2d(in_channels = 6, out_channels = 1, kernel_size=5, padding=2, stride = 2, output_padding = 1)
    

    def forward(self, x, verbose=False):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 16, 14, 14)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_components = 10
encoder = EncoderDeepConv(n_components)
encoder.to(device)

decoder = DecoderDeepConv(n_components)
decoder.to(device)

criterion = nn.MSELoss()
parameters = (list)


    