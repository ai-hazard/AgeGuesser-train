from torch import nn
import timm

def build_backbone(name='tf_efficientnetv2_b0', pretrained=True):
    backbone_ = timm.create_model(name, pretrained=pretrained)

    last_conv_channels = list(backbone_.children())[-1].in_features
    backbone = nn.Sequential(
        *list(backbone_.children())[:-1],
    )
    return backbone, last_conv_channels

class AgeNetwork(nn.Module):
    def __init__(self, backbone_name="tf_efficientnetv2_b0"):
        super(AgeNetwork, self).__init__()
        
        self.backbone, num_ftrs = build_backbone(backbone_name)
        
        self.model1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_ftrs),
            nn.Dropout(0.2),
            nn.LazyLinear(8),
            nn.LazyLinear(1)
        )
          
    def forward(self, x):
        x = self.backbone(x)
        x = self.model1(x)
        return x
    