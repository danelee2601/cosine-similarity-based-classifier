import torch
from torch import nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)
    

class CosineSimClf1D(nn.Module):
    def __init__(self, in_channels:int, num_classes:int, dropout=0.3, init_scale=10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc = nn.Sequential(*[
            ConvBlock(in_channels, 128, 8, 1),
            nn.Dropout(dropout),
            ConvBlock(128, 256, 5, 1),
            nn.Dropout(dropout),
            
            ConvBlock(256, 256, 5, 1),
            nn.Dropout(dropout),
            ConvBlock(256, 256, 5, 1),
            nn.Dropout(dropout),
            ConvBlock(256, 256, 5, 1),
            nn.Dropout(dropout),

            Conv1dSamePadding(256, 128, 3, 1),
        ])

        self.feature_dim = 128
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(init_scale))
        self.class_weights = nn.Parameter(torch.FloatTensor(num_classes, self.feature_dim))  # (n_classes, d)
        nn.init.kaiming_uniform_(self.class_weights)
        
    def forward(self, x: torch.FloatTensor, return_feature_vector:bool=False):
        """
        x: (b c l)
        """
        out = self.enc(x)  # (b d l'); l' < l
        z = out.mean(dim=-1)  # (b d); representation

        if return_feature_vector:
            return z

        # L2 normalize the feature vectors and the class weights
        normalized_features = F.normalize(z, p=2, dim=1)  # (b d)
        normalized_weights = F.normalize(self.class_weights, p=2, dim=1)  # (n_classes, d)

        # Compute cosine similarity
        cosine_similarity = torch.matmul(normalized_features, normalized_weights.t())  # (b d) * (d n_classes) -> (b n_classes)
        
        # Scale the cosine similarity
        logits = scaled_similarity = self.scale * cosine_similarity  # (b n_classes)

        return logits  # (b n_classes)


if __name__ == '__main__':
    batch_size = 4
    in_channels = 1  # indicating univarate time seires
    seq_length = 100
    x = torch.randn(batch_size, in_channels, seq_length)  # (b c l)

    clf = CosineSimClf1D(in_channels, num_classes=10)
    logits = clf(x)  # (4, 10)
    print('logits.shape:', logits.shape)