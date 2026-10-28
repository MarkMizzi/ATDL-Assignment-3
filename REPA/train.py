import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
SATDINO_DEFAULT_MEAN = (0.433, 0.436, 0.407)
SATDINO_DEFAULT_STD = (0.198, 0.193, 0.195)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif enc_type == 'satdino':
        x = x / 255.
        x = Normalize(SATDINO_DEFAULT_MEAN, SATDINO_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def sample_grid_caption(labels, class_names=None):
    nrow = max(1, round(math.sqrt(len(labels))))
    names = [
        f"{label}: {class_names[label]}" if class_names else f"Class {label}"
        for label in labels
    ]
    return '\n'.join(
        f"Row {start // nrow + 1}: " + ' | '.join(names[start:start + nrow])
        for start in range(0, len(names), nrow)
    )


def balanced_subset_indices(labels, size, seed):
    labels = np.asarray(labels)
    size = min(size, len(labels))
    classes = np.unique(labels)
    if size <= 0 or len(classes) == 0:
        raise ValueError("Training evaluation subset and labels must be non-empty")

    rng = np.random.default_rng(seed)
    base_count, remainder = divmod(size, len(classes))
    selected = []
    for class_index, label in enumerate(classes):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        class_count = base_count + (class_index < remainder)
        if len(indices) < class_count:
            raise ValueError(
                f"Training class {label} has {len(indices)} samples; "
                f"{class_count} are needed for an evaluation subset of size {size}."
            )
        selected.extend(indices[:class_count].tolist())
    return sorted(selected)


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


LORA_TARGET_PATTERNS = {
    'qkv': r'attn\.qkv',
    'proj': r'attn\.proj',
    'fc1': r'mlp\.fc1',
    'fc2': r'mlp\.fc2',
    'adaLN_modulation': r'adaLN_modulation\.1',
}


def lora_target_regex(targets):
    # Scoped to SiT blocks: a bare "proj" would also match the x_embedder.proj Conv2d.
    return r'blocks\.\d+\.(' + '|'.join(LORA_TARGET_PATTERNS[t] for t in targets) + r')'


def apply_lora(model, rank, targets, train_io=False):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        bias='none',
        target_modules=lora_target_regex(targets),
    )
    model = get_peft_model(model, config)
    # New class embeddings and alignment projectors must stay fully trainable.
    trainable = ['.y_embedder.', '.projectors.']
    if train_io:
        trainable += ['.x_embedder.', '.final_layer.']
    for name, param in model.named_parameters():
        if any(part in name for part in trainable):
            param.requires_grad = True
    return model


@torch.no_grad()
def merged_lora_state_dict(peft_model):
    """Plain SiT state dict with LoRA deltas merged in, built on CPU."""
    from peft.tuners.lora import LoraLayer

    base = peft_model.get_base_model()
    state = {
        key.replace('.base_layer.', '.'): value.detach().cpu().clone()
        for key, value in base.state_dict().items()
        if '.lora_' not in key
    }
    for name, module in base.named_modules():
        if isinstance(module, LoraLayer):
            weight_key = f'{name}.weight'
            for adapter in module.active_adapters:
                delta = module.get_delta_weight(adapter).detach().cpu()
                state[weight_key] += delta.to(state[weight_key].dtype)
    return state


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != 'none':
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        encoders, encoder_types, architectures = [], [], []
    z_dims = [encoder.embed_dim for encoder in encoders]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    if args.pretrained_path:
        checkpoint = torch.load(
            args.pretrained_path,
            map_location='cpu',
            weights_only=False,
        )
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        state_dict = dict(state_dict)
        label_key = 'y_embedder.embedding_table.weight'
        state_dict[label_key] = model.state_dict()[label_key].detach().clone()
        if args.reset_projectors or args.enc_type == 'none':
            state_dict = {
                key: value for key, value in state_dict.items()
                if not key.startswith('projectors.')
            }
            state_dict.update({
                key: value for key, value in model.state_dict().items()
                if key.startswith('projectors.')
            })
            if args.enc_type == 'none':
                logger.info("Pretrained alignment projectors discarded; alignment disabled.")
            else:
                logger.info("Alignment projectors initialized afresh.")
        else:
            saved_args = checkpoint.get('args')
            saved_enc_type = (
                saved_args.get('enc_type') if isinstance(saved_args, dict)
                else getattr(saved_args, 'enc_type', None)
            )
            saved_enc_names = saved_enc_type.split(',') if saved_enc_type else None
            if saved_enc_names is None:
                logger.warning(
                    "Checkpoint has no encoder metadata; existing projectors are "
                    "matched by index. Keep their original encoder order and variants."
                )
            current_state = model.state_dict()
            for index, enc_name in enumerate(args.enc_type.split(',')):
                prefix = f'projectors.{index}.'
                if saved_enc_names is not None and index < len(saved_enc_names):
                    if saved_enc_names[index] != enc_name:
                        raise ValueError(
                            f"Projector {index} was trained for {saved_enc_names[index]}, "
                            f"not {enc_name}. Keep the pretrained encoder order or "
                            "use --reset-projectors."
                        )
                elif not any(key.startswith(prefix) for key in state_dict):
                    state_dict.update({
                        key: value.detach().clone()
                        for key, value in current_state.items()
                        if key.startswith(prefix)
                    })
                    logger.info(f"New alignment projector initialized for {enc_name}.")
        model.load_state_dict(state_dict, strict=True)
        logger.info(
            f"Loaded pretrained weights from {args.pretrained_path}; "
            "class embeddings initialized afresh."
        )

    if args.lora_rank > 0:
        model = apply_lora(model, args.lora_rank, args.lora_targets, args.lora_train_io)
        if accelerator.is_main_process:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"LoRA rank {args.lora_rank} on {','.join(args.lora_targets)}: "
                f"{trainable_params:,} of {total_params:,} parameters trainable."
            )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] if args.lora_rank > 0
        else model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = None
    train_eval_dataloader = None
    if args.val_dir and accelerator.is_main_process:
        val_dataset = CustomDataset(args.val_dir)
        train_eval_size = min(len(train_dataset), len(val_dataset))
        train_eval_indices = balanced_subset_indices(
            train_dataset.labels, train_eval_size, args.seed + 100003,
        )
        train_eval_dataset = Subset(train_dataset, train_eval_indices)
        train_eval_dataloader = DataLoader(
            train_eval_dataset,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(
            f"Validation uses all {len(val_dataset):,} images "
            f"from {args.val_dir}; train-eval uses a fixed, class-balanced "
            f"subset of {len(train_eval_dataset):,} training images."
        )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
            )
        if args.lora_rank > 0:
            if 'lora_model' not in ckpt or 'lora_ema' not in ckpt:
                raise ValueError(
                    f"Checkpoint {ckpt_name} has no LoRA state; resume it without --lora-rank."
                )
            model.load_state_dict(ckpt['lora_model'])
            ema.load_state_dict(ckpt['lora_ema'])
        else:
            model.load_state_dict(ckpt['model'])
            ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, gt_ys = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_ys = gt_ys[:sample_batch_size].to(device)
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)

    def evaluate_losses(dataloader, split):
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        validation_seed = args.seed + 100003
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_seed)

        denoising_loss_sum = 0.0
        alignment_loss_sum = 0.0
        num_examples = 0
        try:
            ema.eval()
            with torch.no_grad():
                for raw_image, moments, labels in dataloader:
                    raw_image = raw_image.to(device) if encoders else None
                    moments = moments.squeeze(dim=1).to(device)
                    labels = labels.to(device)
                    images = sample_posterior(
                        moments, latents_scale=latents_scale, latents_bias=latents_bias,
                    )
                    zs = []
                    with accelerator.autocast():
                        for encoder, encoder_type in zip(encoders, encoder_types):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                            features = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type:
                                features = features[:, 1:]
                            if 'dinov2' in encoder_type:
                                features = features['x_norm_patchtokens']
                            zs.append(features)
                        denoising_loss, alignment_loss = loss_fn(
                            ema, images, dict(y=labels), zs=zs,
                        )

                    batch_size = images.shape[0]
                    denoising_loss_sum += denoising_loss.sum().item()
                    alignment_loss_sum += alignment_loss.sum().item()
                    num_examples += batch_size
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        return {
            f'{split}/diffusion_loss': denoising_loss_sum / num_examples,
            f'{split}/alignment_loss': alignment_loss_sum / num_examples,
        }

    loss_curve_epochs = []
    train_eval_diffusion_curve = []
    train_eval_alignment_curve = []
    val_curve_epochs = []
    val_diffusion_curve = []
    val_alignment_curve = []

    for epoch in range(args.epochs):
        model.train()
        epoch_complete = True
        epoch_diffusion_loss_sum = 0.0
        epoch_alignment_loss_sum = 0.0
        epoch_total_loss_sum = 0.0
        epoch_example_count = 0
        epoch_grad_norm_sum = 0.0
        epoch_optimizer_steps = 0
        for batch_index, (raw_image, x, y) in enumerate(train_dataloader):
            if encoders:
                raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean * args.proj_coeff
                batch_size = x.shape[0]
                epoch_diffusion_loss_sum += loss_mean.detach().item() * batch_size
                epoch_alignment_loss_sum += proj_loss_mean.detach().item() * batch_size
                epoch_total_loss_sum += loss.detach().item() * batch_size
                epoch_example_count += batch_size
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    epoch_grad_norm_sum += grad_norm.detach().item()
                    epoch_optimizer_steps += 1
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    if args.lora_rank > 0:
                        lora_model = accelerator.unwrap_model(model)
                        # Merged weights keep generate.py and --pretrained-path compatible.
                        checkpoint.update({
                            "model": merged_lora_state_dict(lora_model),
                            "ema": merged_lora_state_dict(ema),
                            "lora_model": lora_model.state_dict(),
                            "lora_ema": ema.state_dict(),
                        })
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (batch_index + 1 == len(train_dataloader)
                    and (epoch + 1) % 10 == 0):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        ema,
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        num_classes=args.num_classes,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                out_labels = accelerator.gather(ys).cpu().tolist()
                gt_labels = accelerator.gather(gt_ys).cpu().tolist()
                if accelerator.is_main_process:
                    accelerator.log({
                        "samples": wandb.Image(
                            array2grid(out_samples),
                            caption=sample_grid_caption(out_labels, args.class_names),
                        ),
                        "gt_samples": wandb.Image(
                            array2grid(gt_samples),
                            caption=sample_grid_caption(gt_labels, args.class_names),
                        ),
                    }, step=global_step)
                logging.info(f"Generating EMA samples after epoch {epoch + 1} done.")

            progress_bar.set_postfix(
                loss=loss_mean.detach().item(),
                proj_loss=proj_loss_mean.detach().item(),
            )

            if global_step >= args.max_train_steps:
                epoch_complete = batch_index + 1 == len(train_dataloader)
                break

        epoch_metrics = torch.tensor(
            [
                epoch_diffusion_loss_sum,
                epoch_alignment_loss_sum,
                epoch_total_loss_sum,
                epoch_example_count,
                epoch_grad_norm_sum,
                epoch_optimizer_steps,
            ],
            dtype=torch.float64,
            device=device,
        )
        epoch_metrics = accelerator.gather(epoch_metrics).view(-1, 6).sum(dim=0)
        (diffusion_loss_sum, alignment_loss_sum, total_loss_sum, example_count,
         grad_norm_sum, optimizer_step_count) = epoch_metrics.tolist()
        epoch_logs = {
            "train/epoch": epoch + 1,
            "train/epoch_complete": int(epoch_complete),
            "train/diffusion_loss": diffusion_loss_sum / example_count,
            "train/alignment_loss": alignment_loss_sum / example_count,
            "train/total_loss": total_loss_sum / example_count,
        }
        if optimizer_step_count > 0:
            epoch_logs["train/grad_norm"] = grad_norm_sum / optimizer_step_count
        accelerator.log(epoch_logs, step=global_step)
        logger.info(
            f"Training after epoch {epoch + 1} at step {global_step}: "
            f"diffusion={epoch_logs['train/diffusion_loss']:.6f}, "
            f"alignment={epoch_logs['train/alignment_loss']:.6f}, "
            f"complete={epoch_complete}"
        )

        if (epoch_complete and val_dataloader is not None
                and (epoch + 1) % args.validation_epochs == 0):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                train_eval_logs = evaluate_losses(train_eval_dataloader, 'train_eval')
                val_logs = evaluate_losses(val_dataloader, 'val')
                loss_curve_epochs.append(epoch + 1)
                train_eval_diffusion_curve.append(
                    train_eval_logs['train_eval/diffusion_loss']
                )
                train_eval_alignment_curve.append(
                    train_eval_logs['train_eval/alignment_loss']
                )
                val_curve_epochs.append(epoch + 1)
                val_diffusion_curve.append(val_logs['val/diffusion_loss'])
                val_alignment_curve.append(val_logs['val/alignment_loss'])
                accelerator.log({
                    **train_eval_logs,
                    **val_logs,
                    "loss_curves/diffusion": wandb.plot.line_series(
                        xs=[loss_curve_epochs, val_curve_epochs],
                        ys=[train_eval_diffusion_curve, val_diffusion_curve],
                        keys=["train-eval", "validation"],
                        title="Diffusion loss",
                        xname="Epoch",
                    ),
                    "loss_curves/alignment": wandb.plot.line_series(
                        xs=[loss_curve_epochs, val_curve_epochs],
                        ys=[train_eval_alignment_curve, val_alignment_curve],
                        keys=["train-eval", "validation"],
                        title="Alignment loss",
                        xname="Epoch",
                    ),
                }, step=global_step)
                logger.info(
                    f"Train-eval/validation after epoch {epoch + 1} at step {global_step}: "
                    f"train-eval diffusion={train_eval_logs['train_eval/diffusion_loss']:.6f}, "
                    f"train-eval alignment={train_eval_logs['train_eval/alignment_loss']:.6f}, "
                    f"diffusion={val_logs['val/diffusion_loss']:.6f}, "
                    f"alignment={val_logs['val/alignment_loss']:.6f}"
                )
            accelerator.wait_for_everyone()

        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Trusted checkpoint for fine-tuning with fresh class embeddings and optimizer.")
    parser.add_argument("--reset-projectors", action="store_true",
                        help="Reset alignment projectors when changing pretrained alignment encoders.")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank for fine-tuning; 0 disables LoRA (full fine-tuning).")
    parser.add_argument("--lora-targets", type=str, default="qkv,proj,fc1,fc2",
                        help=f"Comma-separated SiT block layers for LoRA: {','.join(LORA_TARGET_PATTERNS)}.")
    parser.add_argument("--lora-train-io", action="store_true",
                        help="With LoRA, also fully train the patch embedding and final layer.")

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Optional held-out dataset directory for loss evaluation.")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-epochs", type=int, default=5,
                        help="Run validation after each N completed training epochs.")
    parser.add_argument("--class-names", nargs='+', default=None,
                        help="Names in class-ID order for sample captions; defaults to alphabetical EuroSAT names for 10 classes, otherwise numeric IDs.")

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b',
                        help="Comma-separated alignment encoders, or 'none' for diffusion-only training.")
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.enc_type.lower() == 'none':
        args.enc_type = 'none'
    if args.class_names is None and args.num_classes == 10:
        args.class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake',
        ]
    if args.class_names is not None and len(args.class_names) != args.num_classes:
        parser.error("--class-names must provide one name per class in class-ID order")
    if args.validation_epochs < 1:
        parser.error("--validation-epochs must be positive")
    if args.pretrained_path and args.resume_step > 0:
        parser.error("--pretrained-path and --resume-step cannot be combined")
    if args.reset_projectors and not args.pretrained_path:
        parser.error("--reset-projectors requires --pretrained-path")
    if args.lora_rank < 0:
        parser.error("--lora-rank must be non-negative")
    args.lora_targets = [target.strip() for target in args.lora_targets.split(',') if target.strip()]
    unknown_targets = sorted(set(args.lora_targets) - set(LORA_TARGET_PATTERNS))
    if unknown_targets or not args.lora_targets:
        parser.error(
            f"--lora-targets must be a non-empty subset of {','.join(LORA_TARGET_PATTERNS)}; "
            f"got unknown {','.join(unknown_targets)}"
        )
    if args.lora_rank > 0 and not (args.pretrained_path or args.resume_step > 0):
        parser.error("--lora-rank requires --pretrained-path or --resume-step")
    if args.lora_train_io and args.lora_rank == 0:
        parser.error("--lora-train-io requires --lora-rank")

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
