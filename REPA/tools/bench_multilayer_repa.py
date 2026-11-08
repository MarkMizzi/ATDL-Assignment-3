"""Parameter counts, peak memory and step time of each --repa-config on one GPU.

Mirrors one train.py optimizer step (fp16 autocast, AdamW, grad clipping, EMA update) on
random latents and images, with the real frozen SatDINO-B teacher and SiT-L/2. Weights are
random: memory and speed do not depend on their values.

  python tools/bench_multilayer_repa.py --micro-batch=32 --accum=8 --configs=legacy,0,A,B,C,D,E
"""
import argparse
import gc
import json
import os
import sys
import time
from copy import deepcopy

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator  # noqa: E402
from loss import SILoss  # noqa: E402
from models.sit import SiT_models  # noqa: E402
from repa_configs import build_teacher_targets, get_repa_config  # noqa: E402
from train import preprocess_raw_image, update_ema  # noqa: E402
from utils import SATDINO_MODEL_IDS, SatDINOEncoder  # noqa: E402


def count(params):
    return sum(p.numel() for p in params)


def bench(config_id, encoder, args, accelerator):
    device = accelerator.device
    config = None if config_id == 'legacy' else get_repa_config(config_id)
    kwargs = dict(z_dims=[768]) if config is None else dict(
        z_dims=[768], repa_config=config, repa_z_dim=768, repa_seed=0)
    model = SiT_models['SiT-L/2'](
        input_size=32, num_classes=10, use_cfg=True, encoder_depth=8,
        fused_attn=True, qk_norm=False, **kwargs).to(device)
    heads = model.projectors if config is None else model.repa_projectors
    projector_params = count(heads.parameters())
    backbone_params = count(p for p in model.parameters() if p.requires_grad) - projector_params
    ema = deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
    model, optimizer = accelerator.prepare(model, optimizer)
    loss_fn = SILoss(repa_config=config)

    def step():
        for _ in range(args.accum):
            x = torch.randn(args.micro_batch, 4, 32, 32, device=device)
            y = torch.randint(0, 10, (args.micro_batch,), device=device)
            raw = torch.randint(0, 256, (args.micro_batch, 3, 256, 256), device=device).float()
            with torch.no_grad(), accelerator.autocast():
                image = preprocess_raw_image(raw, 'satdino')
                if config is None:
                    zs = [encoder.forward_features(image)]
                else:
                    zs = build_teacher_targets(config, encoder.forward_layers(image, config.teacher_layers))
            with accelerator.autocast():
                denoising, proj = loss_fn(model, x, dict(y=y), zs=zs)
                loss = (denoising.mean() + 0.5 * proj.mean()) / args.accum
            accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema, model)
        return denoising.mean().item(), proj.mean().item()

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):
        losses = step()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        losses = step()
    torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - start) / args.steps
    result = dict(
        config=config_id,
        projectors=len(heads),
        projector_params=projector_params,
        backbone_trainable_params=backbone_params,
        total_trainable_params=projector_params + backbone_params,
        adam_state_gb=2 * 4 * (projector_params + backbone_params) / 1e9,
        ema_gb=4 * count(ema.parameters()) / 1e9,
        peak_mem_gb=torch.cuda.max_memory_allocated(device) / 1e9,
        sec_per_opt_step=seconds,
        effective_batch=args.micro_batch * args.accum,
        finite=all(map(lambda v: v == v and abs(v) < float('inf'), losses)),
        diffusion_loss=losses[0],
        alignment_loss=losses[1],
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='legacy,S8T12,S8T9,A,B,C,D,E')
    parser.add_argument('--micro-batch', type=int, default=32)
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    accelerator = Accelerator(mixed_precision='fp16')
    encoder = SatDINOEncoder(SATDINO_MODEL_IDS['b']).to(accelerator.device)
    results = []
    for config_id in args.configs.split(','):
        try:
            result = bench(config_id, encoder, args, accelerator)
        except torch.OutOfMemoryError:
            result = dict(config=config_id, error=f'OOM at micro-batch {args.micro_batch}')
        # prepare() keeps references to the model and optimizer; drop them between configs.
        accelerator.free_memory()
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(result), flush=True)
        results.append(result)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(device=torch.cuda.get_device_name(), args=vars(args), results=results), f, indent=2)


if __name__ == '__main__':
    main()
