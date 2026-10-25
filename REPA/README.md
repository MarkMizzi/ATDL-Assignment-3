<h1 align="center"> Representation Alignment for Generation: <br>Training Diffusion Transformers Is Easier Than You Think
</h1>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.06940-b31b1b.svg)](https://arxiv.org/abs/2410.06940)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-alignment-for-generation/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=representation-alignment-for-generation)

<div align="center">
  <a href="https://sihyun.me/" target="_blank">Sihyun&nbsp;Yu</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://www.linkedin.com/in/SangkyungKwak/" target="_blank">Sangkyung&nbsp;Kwak</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://huiwon-jang.github.io/" target="_blank">Huiwon&nbsp;Jang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://jh-jeong.github.io/" target="_blank">Jongheon&nbsp;Jeong</a><sup>2</sup>
  <br>
  <a href="http://jonathan-huang.org/" target="_blank">Jonathan&nbsp;Huang</a><sup>3</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://alinlab.kaist.ac.kr/shin.html" target="_blank">Jinwoo&nbsp;Shin</a><sup>1*</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://www.sainingxie.com/" target="_blank">Saining&nbsp;Xie</a><sup>4*</sup><br>
  <sup>1</sup> KAIST &emsp; <sup>2</sup>Korea University &emsp; <sup>3</sup>Scaled Foundations &emsp; <sup>4</sup>New York University &emsp; <br>
  <sup>*</sup>Equal Advising &emsp; <br>
</div>
<h3 align="center">[<a href="https://sihyun.me/REPA">project page</a>]&emsp;[<a href="http://arxiv.org/abs/2410.06940">arXiv</a>]</h3>
<br>

<b>Summary</b>: We propose REPresentation Alignment (REPA), a method that aligns noisy input states in diffusion models with representations from pretrained visual encoders. This significantly improves training efficiency and generation quality. REPA speeds up SiT training by 17.5x and achieves state-of-the-art FID=1.42.

### 1. Environment setup

```bash
conda create -n repa python=3.9 -y
conda activate repa
pip install -r requirements.txt
```

### 2. Dataset

#### Dataset download

Currently, we provide experiments for [ImageNet](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). You can place the data that you want and can specifiy it via `--data-dir` arguments in training scripts. Please refer to our [preprocessing guide](https://github.com/sihyun-yu/REPA/tree/master/preprocessing).

For EuroSAT, create reproducible 70/15/15 train/validation/test splits of paired
preprocessed images and latents with the
[split script](preprocessing/split_eurosat.py).
Train only on the resulting `train` directory. Validation should be used for
checkpoint selection and tuning; pass the resulting `val` directory with
`--val-dir`. The script logs diffusion and alignment losses on the full
validation split every five completed training epochs by default; adjust the
interval with `--validation-epochs`. At each validation, it also evaluates a
fixed, class-balanced training subset of the same size using EMA weights in eval
mode. W&B charts compare these train-eval losses with validation losses; ordinary
online training losses are still averaged and logged once per epoch. EMA sample
grids are generated every ten completed epochs. Keep `test` untouched until
final evaluation.

### 3. Training

```bash
accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="linear-dinov2-b-enc8" \
  --data-dir=[YOUR_DATA_PATH]
```

Then this script will automatically create the folder in `exps` to save logs and checkpoints. You can adjust the following options:

- `--models`: `[SiT-B/2, SiT-L/2, SiT-XL/2]`
- `--enc-type`: `[dinov2-vit-b, dinov2-vit-l, dinov2-vit-g, dinov1-vit-b, mocov3-vit-b, , mocov3-vit-l, clip-vit-L, jepa-vit-h, mae-vit-l]`
- `--proj-coeff`: Any values larger than 0
- `--encoder-depth`: Any values between 1 to the depth of the model
- `--output-dir`: Any directory that you want to save checkpoints and logs
- `--exp-name`: Any string name (the folder will be created under `output-dir`)

For DINOv2 models, it will be automatically downloaded from `torch.hub`. For CLIP models, it will be also automatically downloaded from the CLIP repository. For other pretrained visual encoders, please download the model weights from the below links and place into the following directories with these names:

- `dinov1`: Download the ViT-B/16 model from the [`DINO`](https://github.com/facebookresearch/dino) repository and place it as `./ckpts/dinov1_vitb.pth`
- `mocov3`: Download the ViT-B/16 or ViT-L/16 model from the [`RCG`](https://github.com/LTH14/rcg) repository and place them as `./ckpts/mocov3_vitb.pth` or `./ckpts/mocov3_vitl.pth`
- `jepa`: Download the ViT-H/14 model (ImageNet-1K) from the [`I-JEPA`](https://github.com/facebookresearch/ijepa) repository and place it as `./ckpts/ijepa_vith.pth`
- `mae`: Download the ViT-L model from [`MAE`](https://github.com/facebookresearch/mae) repository and place it as `./ckpts/mae_vitl.pth`

#### Sample class captions

W&B captions for `samples` and `gt_samples` list class IDs and names row by row,
from left to right in each grid. Labels are gathered in the same process order
as images in distributed runs. For `--num-classes=10`, the default names follow
alphabetical EuroSAT class-folder order: AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River, SeaLake.

If your label mapping differs, pass `--class-names` followed by one name per
class in ID order (quote names containing spaces). Other class counts use
numeric IDs unless names are supplied. Names affect captions only, not training.

#### Training without alignment

Pass `--enc-type=none` to `train.py` for diffusion-only training. No alignment
encoder is loaded and no alignment projectors are created. The alignment loss
is zero and `--proj-coeff` has no effect. Class conditioning is unchanged.

This works both from scratch and with `--pretrained-path=/path/to/checkpoint.pt`.
When fine-tuning a REPA checkpoint, its alignment projector weights are discarded
automatically while the SiT backbone is retained. No `--reset-projectors` flag
is needed. Resume diffusion-only checkpoints with `--enc-type=none` as well;
use pretrained loading, not `--resume-step`, to switch an aligned run to this mode.

#### SatDINO alignment on EuroSAT

For RGB EuroSAT training with `train.py`, use the frozen
[`strakajk/satdino-vit_base-16`](https://huggingface.co/strakajk/satdino-vit_base-16)
encoder instead of DINOv2:

```bash
--enc-type=satdino-vit-b --resolution=256 --num-classes=10
```

Or align to both encoders with separate learned projectors:

```bash
--enc-type=dinov2-vit-l,satdino-vit-b --resolution=256 --num-classes=10
```

The existing loss averages the two alignment losses equally; `--proj-coeff`
weights their combined contribution. SatDINO uses fMoW RGB normalization and
256x256 inputs, producing 256 patch tokens of width 768 for SiT `/2` models.
Its class and trailing GSD register tokens are excluded. DINOv2 preprocessing
is unchanged. SatDINO alignment is currently supported only at resolution 256.

The first run downloads the encoder through Transformers with
`trust_remote_code=True`, which executes code from the model repository. Review
and trust that repository before using this option. Transformers is already
listed in the requirements.

SatDINO is constructed from its configuration, then its published
`pytorch_model.bin` weights are loaded with PyTorch using `weights_only=True`
and strict key checking. This bypasses the Transformers pretrained-loading
path that fails with `all_tied_weights_keys` on this older custom model under
Transformers 5. No dependency downgrade is required by this loading path.

When adding SatDINO to a DINOv2-only ImageNet REPA checkpoint, use
`--pretrained-path=/path/to/checkpoint.pt` and keep the same DINOv2 variant first
in `--enc-type`. The pretrained DINOv2 projector is retained and only the new
SatDINO projector is initialized afresh. Do not pass `--reset-projectors` in
this case. Saved encoder metadata is checked when available; checkpoints without
metadata are matched by projector index, so their original encoder order and
variants must be preserved.

To replace DINOv2 with SatDINO alone or change an existing encoder selection,
add `--reset-projectors`. This initializes all alignment projectors afresh
while retaining the pretrained SiT backbone. Do not use this flag when resuming
the same run. Only SatDINO ViT-B/16 is supported.

**[12/17/2024]**: We also support training on 512x512 resolution (ImageNet) and a text-to-image generation on MS-COCO.

For ImageNet 512x512, please use the following script:

```bash
accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-XL/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="linear-dinov2-b-enc8-in512" \
  --resolution=512 \
  --data-dir=[YOUR_DATA_PATH]
```

You also need a new data preprocessing that resizes each image to 512x512 resolution and encodes each image as 64x64 resolution latent vectors (using stable-diffusion VAE). This script is also provided in our preprocessing guide.

For text-to-image generation, please follow the data preprocessing protocol in [U-ViT](https://github.com/baofff/U-ViT/tree/main/scripts) before lanuching experiments. After that, you should be able to lanuch an experiment through the following script:

```bash
accelerate launch train_t2i.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="t2i_repa" \
  --data-dir=[YOUR_DATA_PATH]
```


### 4. Evaluation

You can generate images (and the .npz file can be used for [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) suite) through the following script:

```bash
torchrun --nnodes=1 --nproc_per_node=8 generate.py \
  --model SiT-XL/2 \
  --num-fid-samples 50000 \
  --ckpt YOUR_CHECKPOINT_PATH \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.8 \
  --guidance-high=0.7
```

We also provide the SiT-XL/2 checkpoint (trained for 4M iterations) used in the final evaluation. It will be automatically downloaded if you do not specify `--ckpt`.

### Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to carry out sanity-check experiments in the near future.

## Acknowledgement

This code is mainly built upon [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), [edm2](https://github.com/NVlabs/edm2), and [RCG](https://github.com/LTH14/rcg) repositories.\
We also appreciate [Kyungmin Lee](https://kyungmnlee.github.io/) for providing the initial version of the implementation.

## BibTeX

```bibtex
@inproceedings{yu2025repa,
  title={Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think},
  author={Sihyun Yu and Sangkyung Kwak and Huiwon Jang and Jongheon Jeong and Jonathan Huang and Jinwoo Shin and Saining Xie},
  year={2025},
  booktitle={International Conference on Learning Representations},
}
```
