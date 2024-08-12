import torch
import torch.nn.functional as F

class CNNImageClassifier(torch.nn.Module):
    def __init__(self):
        super(CNNImageClassifier, self).__init__()
        conv_layers_kwargs = {
            'kernel_size': 3,
            'stride': 2,
            'padding': 0,
            'dilation': 1,
            'bias': True
        }
        # TODO: rewrite this better
        # TODO: add max pooling, batch normalization, dropout
        # TODO: add choice for nbr of CNN & FF layers (with size-preserving convolutions)?
        # TODO: add skip connections?
        self.conv_layers = [
            torch.nn.Conv2d(in_channels=3, out_channels=48, **conv_layers_kwargs),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=48, out_channels=128, **conv_layers_kwargs),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=192, **conv_layers_kwargs),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=192, out_channels=192, **conv_layers_kwargs),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=192, out_channels=128, **conv_layers_kwargs),
            torch.nn.ReLU(),
        ]
        self.FF_layers = [
            torch.nn.Linear(7 ** 2 * 128, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 999)
        ]
        self.convolutions = torch.nn.Sequential(*self.conv_layers)
        self.FF = torch.nn.Sequential(*self.FF_layers)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.reshape((1, -1))
        y = self.FF(x)
        return y