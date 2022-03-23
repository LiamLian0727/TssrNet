import torch
import torchvision
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator

# load a pre-trained model for classification and return only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FCOS needs to know the number of output channels in a backbone.
# For mobilenet_v2, it's 1280 so we need to add it here
backbone.out_channels = 1280

# let's make the network generate 5 x 3 anchors per spatial location,
# with 5 different sizes and 3 different aspect ratios.
# We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios
anchor_generator = AnchorGenerator(
    sizes=((8,), (16,), (32,), (64,), (128,)),
    aspect_ratios=((1.0,),)
)
# put the pieces together inside a FCOS model
model = FCOS(backbone,
             num_classes=80,
             anchor_generator=anchor_generator,
             )
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400),torch.rand(3, 500, 400),torch.rand(3, 500, 400),torch.rand(3, 500, 400)]
predictions = model(x)

