import torch.nn
import torch.nn.functional as F

class CNNImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers_kwargs = {
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'dilation': 1,
            'bias': True
        }
        # TODO: add batch normalization, dropout
        self.conv_layers = [
            torch.nn.Conv2d(in_channels=3, out_channels=48, **conv_layers_kwargs),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=48, out_channels=128, **conv_layers_kwargs),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=192, **conv_layers_kwargs),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=192, out_channels=192, **conv_layers_kwargs),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=192, out_channels=128, **conv_layers_kwargs),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.FF_layers = [
            torch.nn.Linear(8 ** 2 * 128, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            # MODIFICATION FOR DEV PURPOSES -- CHANGE THIS
            torch.nn.Linear(2048, 999)
            #torch.nn.Linear(2048, 8)
        ]
        self.convolutions = torch.nn.Sequential(*self.conv_layers)
        self.FF = torch.nn.Sequential(*self.FF_layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.convolutions(x)
        x = x.reshape((batch_size, -1))
        y = self.FF(x)
        return y