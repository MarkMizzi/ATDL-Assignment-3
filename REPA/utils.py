import os
from torchvision.datasets.utils import download_url
import torch
import torchvision.models as torchvision_models
import timm
from models import mocov3_vit
import math
import warnings


# code from SiT repository
pretrained_models = {'last.pt'}

def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model

def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict

SATDINO_MODEL_IDS = {
    's': 'strakajk/satdino-vit_small-16',
    'b': 'strakajk/satdino-vit_base-16',
}


class SatDINOEncoder(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        weights_path = hf_hub_download(model_id, filename='pytorch_model.bin')
        state_dict = torch.load(
            weights_path, map_location='cpu', weights_only=True,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.embed_dim = self.model.embed_dim
        self.requires_grad_(False)
        self.eval()

    def forward_features(self, images):
        return self.model(images, return_all=True)[:, 1:-1]

    def forward_layers(self, images, layers):
        """Patch tokens after the given 1-indexed blocks from one forward pass.

        Each layer goes through the frozen final norm, as in SatDINO's own
        forward_intermediate_layers, so the last block equals forward_features.
        Class and trailing GSD register tokens are excluded.
        """
        blocks = self.model.blocks
        layers = sorted(set(layers))
        if not layers or layers[0] < 1 or layers[-1] > len(blocks):
            raise ValueError(f"SatDINO layers must be in 1..{len(blocks)}, got {layers}")
        x = self.model.prepare_tokens(images)
        features = {}
        for index, block in enumerate(blocks, start=1):
            x = block(x)
            if index in layers:
                features[index] = self.model.norm(x)[:, 1:-1]
            if index == layers[-1]:
                break
        return features


# Teachers supported by --repa-config, and their block counts by model size
# (MAE is always ViT-L and I-JEPA always ViT-H in load_encoders).
REPA_TEACHER_DEPTHS = {
    'satdino': {'s': 12, 'b': 12},
    'dinov2': {'s': 12, 'b': 12, 'l': 24, 'g': 40},
    'mocov3': {'s': 12, 'b': 12, 'l': 24},
    'clip': {'b': 12, 'l': 24},
    'mae': {'l': 24},
    'jepa': {'h': 32},
}


def teacher_depth(enc_name):
    """Number of transformer blocks of a --enc-type teacher, without loading it."""
    encoder_type, architecture, model_config = enc_name.split('-')
    family = 'dinov2' if 'dinov2' in encoder_type else encoder_type
    if family == 'mae':
        model_config = 'l'
    elif family == 'jepa':
        model_config = 'h'
    depths = REPA_TEACHER_DEPTHS.get(family, {})
    if architecture != 'vit' or model_config.lower() not in depths:
        raise ValueError(f"--repa-config does not support --enc-type={enc_name}; supported: "
                         + ', '.join(f'{f}-vit-{c}' for f, cs in REPA_TEACHER_DEPTHS.items() for c in cs))
    return depths[model_config.lower()]


def baseline_teacher_target(encoder, encoder_type, images):
    """The single-layer REPA target, exactly as train.py builds it from forward_features."""
    z = encoder.forward_features(images)
    if 'mocov3' in encoder_type: z = z[:, 1:]
    if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
    return z


def _teacher_blocks(encoder, encoder_type):
    """(blocks, to_target): the teacher's transformer blocks, and the map from a block's
    output to the baseline target layout (final norm and token selection as in
    baseline_teacher_target), so the last block gives exactly the baseline target."""
    if encoder_type == 'satdino':
        model = encoder.model
        return model.blocks, lambda x: model.norm(x)[:, 1:-1]  # drop class and GSD tokens
    if 'dinov2' in encoder_type:
        blocks = encoder.blocks
        if getattr(encoder, 'chunked_blocks', False):
            blocks = [b for chunk in blocks for b in chunk if not isinstance(b, torch.nn.Identity)]
        skip = 1 + encoder.num_register_tokens  # class and register tokens
        return blocks, lambda x: encoder.norm(x)[:, skip:]
    if encoder_type == 'mocov3':
        return encoder.blocks, lambda x: encoder.norm(x)[:, 1:]
    if encoder_type == 'mae':  # its forward_features skips the final norm
        return encoder.blocks, lambda x: x[:, 1:]
    if encoder_type == 'jepa':  # no class token
        return encoder.blocks, lambda x: x if encoder.norm is None else encoder.norm(x)
    if encoder_type == 'clip':  # sequence-first blocks, no final norm
        return encoder.model.transformer.resblocks, lambda x: x.permute(1, 0, 2)[:, 1:]
    raise NotImplementedError(f"--repa-config does not support {encoder_type} teachers")


def num_teacher_blocks(encoder, encoder_type):
    return len(_teacher_blocks(encoder, encoder_type)[0])


def forward_teacher_layers(encoder, encoder_type, images, layers):
    """{1-indexed teacher layer: [B, N, D]} from one forward pass; each layer is
    post-processed like the baseline target (see _teacher_blocks)."""
    blocks, to_target = _teacher_blocks(encoder, encoder_type)
    layers = sorted(set(layers))
    if not layers or layers[0] < 1 or layers[-1] > len(blocks):
        raise ValueError(f"Teacher layers must be in 1..{len(blocks)}, got {layers}")
    if hasattr(encoder, 'forward_layers'):  # SatDINO: stops after the last requested block
        return encoder.forward_layers(images, layers)
    captured = {}
    hooks = [blocks[i - 1].register_forward_hook(
        lambda module, inputs, output, i=i: captured.__setitem__(i, output)) for i in layers]
    try:
        encoder.forward_features(images)
    finally:
        for hook in hooks:
            hook.remove()
    return {i: to_target(captured[i]) for i in layers}


def check_teacher_layers(encoder, encoder_type, images):
    """Raise unless the final teacher layer equals the baseline REPA target."""
    final = num_teacher_blocks(encoder, encoder_type)
    layer = forward_teacher_layers(encoder, encoder_type, images, [final])[final]
    target = baseline_teacher_target(encoder, encoder_type, images)
    if layer.shape != target.shape or not torch.allclose(layer.float(), target.float(), atol=1e-4, rtol=1e-3):
        raise RuntimeError(f"{encoder_type}: final teacher layer {tuple(layer.shape)} does not match the "
                           f"baseline REPA target {tuple(target.shape)}")


@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'satdino':
            if architecture != 'vit' or model_config not in SATDINO_MODEL_IDS:
                raise ValueError("SatDINO supports --enc-type=satdino-vit-s or satdino-vit-b")
            encoder = SatDINOEncoder(SATDINO_MODEL_IDS[model_config]).to(device)

        elif encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder = encoder.to(device)
            encoder.eval()

        elif 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        
        elif 'dinov1' == encoder_type:
            import timm
            from models import dinov1
            encoder = dinov1.vit_base()
            ckpt =  torch.load(f'./ckpts/dinov1_vit{model_config}.pth') 
            if 'pos_embed' in ckpt.keys():
                ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    ckpt['pos_embed'], [16, 16],
                )
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder.load_state_dict(ckpt, strict=True)
            encoder = encoder.to(device)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'clip':
            import clip
            from models.clip_vit import UpdatedVisionTransformer
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu')[0].visual
            encoder = UpdatedVisionTransformer(encoder_).to(device)
             #.to(device)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from models.mae_vit import vit_large_patch16
            import timm
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs).to(device)
            with open(f"ckpts/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )

        elif encoder_type == 'jepa':
            from models.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            with open(f"ckpts/ijepa_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f, map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def load_legacy_checkpoints(state_dict, encoder_depth):
    new_state_dict = dict()
    for key, value in state_dict.items():
        if 'decoder_blocks' in key:
            parts =key.split('.')
            new_idx = int(parts[1]) + encoder_depth
            parts[0] = 'blocks'
            parts[1] = str(new_idx)
            new_key = '.'.join(parts)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict