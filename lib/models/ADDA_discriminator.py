"""Discriminator model for ADDA."""
import torch
import torch.nn as nn


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2),
            nn.LogSoftmax()
        )
        self.layer.apply(init_weights)

    def forward(self, input):
        """Forward the discriminator."""
        #print("Discriminator input forward: ", input.size())        
        input = self.avgpool(input)
        #print("Discriminator input forward avgpool: ", input.size())  
        input = torch.flatten(input, 1)
        #print("Discriminator input forward flatten: ", input.size())
        out = self.layer(input)
        return out
      
def get_discriminator():
   
    model = Discriminator()

    return model