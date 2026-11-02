"""Frozen image feature extractors for evaluating generated EuroSAT samples."""
import os

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from utils import SATDINO_MODEL_IDS, SatDINOEncoder

ENCODERS = ('satdino-b', 'satdino-s', 'ssl4eo-moco', 'sscd')

# fMoW RGB statistics, identical to train.py's SatDINO preprocessing.
SATDINO_MEAN = (0.433, 0.436, 0.407)
SATDINO_STD = (0.198, 0.193, 0.195)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SSCD_URL = 'https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt'


class FeatureEncoder(torch.nn.Module):
    """Maps uint8 NHWC image batches to one global feature vector per image."""

    def __init__(self, model, mean, std, size=None):
        super().__init__()
        self.model = model
        self.normalize = Normalize(mean, std)
        self.size = size

    @torch.no_grad()
    def forward(self, images):
        x = images.permute(0, 3, 1, 2).float() / 255.
        if self.size is not None and x.shape[-2:] != (self.size, self.size):
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', antialias=True, align_corners=False)
        return self.model(self.normalize(x))


def load_sscd(device):
    path = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(SSCD_URL))
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(SSCD_URL, path)
    return torch.jit.load(path, map_location=device)


def load_encoder(name, device):
    if name.startswith('satdino-'):
        # SatDINOModel.forward returns the CLS token.
        model = SatDINOEncoder(SATDINO_MODEL_IDS[name.split('-')[1]]).model
        encoder = FeatureEncoder(model, SATDINO_MEAN, SATDINO_STD)
    elif name == 'ssl4eo-moco':
        from torchgeo.models import ResNet50_Weights, resnet50

        # num_classes=0 makes timm return globally pooled 2048-d features.
        model = resnet50(ResNet50_Weights.SENTINEL2_RGB_MOCO, num_classes=0)
        encoder = FeatureEncoder(model, (0., 0., 0.), (1., 1., 1.), size=224)
    elif name == 'sscd':
        encoder = FeatureEncoder(load_sscd(device), IMAGENET_MEAN, IMAGENET_STD, size=288)
    else:
        raise ValueError(f"Unknown encoder {name}; choose from {', '.join(ENCODERS)}")
    return encoder.to(device).eval().requires_grad_(False)
