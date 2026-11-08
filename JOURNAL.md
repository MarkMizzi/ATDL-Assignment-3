# Journal

## Dataset Preparation

- 22/10 Mark. Downloaded the EuroSAT dataset with:
  ```python
  import kagglehub 
  kagglehub.login()
  path = kagglehub.dataset_download("apollo2506/eurosat-dataset", output_dir="data")
  ```

- 23/10 Mark. Converted EuroSAT dataset to format used by REPA scripts using
  ```bash
  python REPA/preprocessing/dataset_tools.py convert --source EuroSAT-data/EuroSAT/ --dest images --resolution 256x256 --transform=center-crop-dhariwal
  ```

- 23/10 Mark. Generated EuroSAT latents on UCloud server using
  ```bash
  python REPA/preprocessing/dataset_tools.py encode --source=images --dest=vae-sd
  ```
  Version of REPA code used is latest of personal fork.

- 24/10 Natasha. Implemented the split of EuroSAT dataset into train/validation/test in the proportion 70/15/15. The script for splitting the data has been called as
  ```bash
  python preprocessing/split_eurosat.py --source=/work/data --dest=/work/data/eurosat-splits
  ```
## Training

- 24/10 Natasha. Changed the training script to be able to choose which alignment encoder to use (also none or both DINO and SatDINO are possible). The loss(es) are computed now separately for train
    and validation datasets, and the generated example pictures are saved to WandB more often. The plan is to do the following 4 training schemes (note that --reset-projectors is ONLY used when switching from one alignment encoder to another one!):

  finetune with DINO:
  ```bash
  accelerate launch train.py   \
    --report-to="wandb"        \
    --allow-tf32               \
    --mixed-precision="fp16"   \
    --seed=0                   \
    --path-type="linear"       \
    --prediction="v"           \
    --weighting="uniform"      \
    --model="SiT-L/2"          \
    --enc-type="dinov2-vit-l"  \
    --proj-coeff=0.5           \
    --encoder-depth=8          \
    --output-dir="exps"        \
    --num-classes=10 \
    --resolution=256 \
    --exp-name=eurosat-finetune-dino-sit-l-2 \
    --pretrained-path=./0400000.pt \
    --learning-rate=1e-5 \
    --data-dir=/work/data/eurosat-splits/train \
    --val-dir=/work/data/eurosat-splits/val \
    --checkpointing-steps=730
  ```

  finetune with SatDINO:

  ```bash
  accelerate launch train.py   \
    --report-to="wandb"        \
    --allow-tf32               \
    --mixed-precision="fp16"   \
    --seed=0                   \
    --path-type="linear"       \
    --prediction="v"           \
    --weighting="uniform"      \
    --model="SiT-L/2"          \
    --enc-type="satdino-vit-b"  \
    --proj-coeff=0.5           \
    --encoder-depth=8          \
    --output-dir="exps"        \
    --num-classes=10 \
    --resolution=256 \
    --exp-name=eurosat-finetune-satdino-sit-l-2 \
    --pretrained-path=./0400000.pt \
    --learning-rate=1e-5 \
    --reset-projectors \
    --data-dir=/work/data/eurosat-splits/train \
    --val-dir=/work/data/eurosat-splits/val \
    --checkpointing-steps=730
  ```

  finetune with DINO+SatDINO

  ```bash
  accelerate launch train.py   \
    --report-to="wandb"        \
    --allow-tf32               \
    --mixed-precision="fp16"   \
    --seed=0                   \
    --path-type="linear"       \
    --prediction="v"           \
    --weighting="uniform"      \
    --model="SiT-L/2"          \
    --enc-type="dinov2-vit-l,satdino-vit-b"  \
    --proj-coeff=0.5           \
    --encoder-depth=8          \
    --output-dir="exps"        \
    --num-classes=10 \
    --resolution=256 \
    --exp-name=eurosat-finetune-dino-satdino-sit-l-2 \
    --pretrained-path=./0400000.pt \
    --learning-rate=1e-5 \
    --data-dir=/work/data/eurosat-splits/train \
    --val-dir=/work/data/eurosat-splits/val \
    --checkpointing-steps=730
  ```

  finetune with none:

  ```bash
  accelerate launch train.py   \
    --report-to="wandb"        \
    --allow-tf32               \
    --mixed-precision="fp16"   \
    --seed=0                   \
    --path-type="linear"       \
    --prediction="v"           \
    --weighting="uniform"      \
    --model="SiT-L/2"          \
    --enc-type="none"  \
    --proj-coeff=0.5           \
    --encoder-depth=8          \
    --output-dir="exps"        \
    --num-classes=10 \
    --resolution=256 \
    --exp-name=eurosat-finetune-none-sit-l-2 \
    --pretrained-path=./0400000.pt \
    --learning-rate=1e-5 \
    --data-dir=/work/data/eurosat-splits/train \
    --val-dir=/work/data/eurosat-splits/val \
    --checkpointing-steps=730
  ```
  
- 28/10 Natasha. Added an option for LoRA finetuning instead of full finetuning. Example of calling it:

```bash
  accelerate launch train.py   \
    --report-to="wandb"        \
    --allow-tf32               \
    --mixed-precision="fp16"   \
    --seed=0                   \
    --path-type="linear"       \
    --prediction="v"           \
    --weighting="uniform"      \
    --model="SiT-L/2"          \
    --enc-type="satdino-vit-b"  \
    --proj-coeff=0.5           \
    --encoder-depth=8          \
    --output-dir="exps"        \
    --num-classes=10 \
    --resolution=256 \
    --exp-name=eurosat-finetune-lora-adaln-io-sit-l-2-satdino-b \
    --pretrained-path=./0400000.pt \
    --data-dir=/work/data/eurosat-splits/train \
    --val-dir=/work/data/eurosat-splits/val \
    --checkpointing-steps=730 \
    --learning-rate=1e-4 \
    --reset-projectors \
    --lora-rank=16 \
    --lora-targets=qkv,proj,fc1,fc2,adaLN_modulation \
    --lora-train-io
```

- 28/10 Natasha. Added an option of using satdino-vit-b encoder

## Generation

- 31/10 Mark. To generate images, used commands like

```bash
torchrun \
  --nnodes=1                                                                 \
  --nproc_per_node=1                                                         \
  generate.py                                                                \
  --model SiT-L/2                                                            \
  --num-fid-samples 4000                                                     \
  --ckpt exps/eurosat-from-scratch-dinov2-b-sit-l-2/checkpoints/0060590.pt   \
  --mode sde                                                                 \
  --cfg-scale=1                                                              \
  --projector-embed-dims=768                                                 \
  --num-steps=250                                                            \
  --num-classes=10                                                           \
  --sample-dir=exps/eurosat-from-scratch-dinov2-b-sit-l-2/
```

- 2/11 Mark. To generate images for models trained without REPA:

```bash
torchrun \
  --nnodes=1          \
  --nproc_per_node=1  \
  generate.py         \
  --model SiT-L/2     \
  --num-fid-samples 4000 \
  --ckpt exps/eurosat-from-scratch-norepa-sit-l-2/checkpoints/0060590.pt   \
  --mode sde          \
  --cfg-scale=1       \
  --projector-embed-dims=''    \
  --num-steps=250     \
  --num-classes=10    \
  --sample-dir=exps/eurosat-from-scratch-norepa-sit-l-2/
```

- 07/11 Frederik. Added multi-layer REPA (`--repa-config`, defined in `REPA/repa_configs.py`). Without `--repa-config` nothing changes: the commands above still align one student block (`--encoder-depth`) with the teacher's final layer. With `--repa-config`, the student/teacher layers come from the config, so `--encoder-depth` must be left out. Presets `A`-`E` are the configurations in the report (A: all students -> T12, B: all students -> T9, C: all-to-all, D: all students -> T_mean, E: S_mean -> T_mean); `S<j>T<i>` aligns a single student block j with a single teacher block i (e.g. `S8T9`; `S8T12` equals the original depth-8 run). The presets are defined for a 12-block teacher such as SatDINO-B; a single pair works with any ViT teacher (`satdino`, `dinov2`, `mocov3`, `clip`, `mae`, `jepa`), e.g. `S8T18` with `--enc-type=dinov2-vit-l`. `--reset-projectors` is needed because the heads start fresh.

  finetune with a non-final teacher layer (student block 8 -> SatDINO block 9):

  ```bash
  accelerate launch train.py   \
    --report-to="wandb"        \
    --allow-tf32               \
    --mixed-precision="fp16"   \
    --seed=0                   \
    --path-type="linear"       \
    --prediction="v"           \
    --weighting="uniform"      \
    --model="SiT-L/2"          \
    --enc-type="satdino-vit-b"  \
    --proj-coeff=0.5           \
    --repa-config=S8T9         \
    --output-dir="exps"        \
    --num-classes=10 \
    --resolution=256 \
    --exp-name=eurosat-finetune-satdino-b-mlS8T9-sit-l-2 \
    --pretrained-path=./0400000.pt \
    --learning-rate=1e-5 \
    --reset-projectors \
    --data-dir=/work/data/eurosat-splits/train \
    --val-dir=/work/data/eurosat-splits/val \
    --checkpointing-steps=730
  ```

  finetune with a multi-layer configuration (C: all 6 student blocks aligned with all 4 teacher blocks, 24 projectors):

  ```bash
  accelerate launch train.py   \
    --report-to="wandb"        \
    --allow-tf32               \
    --mixed-precision="fp16"   \
    --seed=0                   \
    --path-type="linear"       \
    --prediction="v"           \
    --weighting="uniform"      \
    --model="SiT-L/2"          \
    --enc-type="satdino-vit-b"  \
    --proj-coeff=0.5           \
    --repa-config=C            \
    --output-dir="exps"        \
    --num-classes=10 \
    --resolution=256 \
    --exp-name=eurosat-finetune-satdino-b-mlC-sit-l-2 \
    --pretrained-path=./0400000.pt \
    --learning-rate=1e-5 \
    --reset-projectors \
    --data-dir=/work/data/eurosat-splits/train \
    --val-dir=/work/data/eurosat-splits/val \
    --checkpointing-steps=730
  ```

## Evaluation

- 04/11 Natasha. Made a faster version of nn_test_for_overfitting that is computing per-pixel differences. It also compares generated pictures both to train and test ones. It can be called as e.g.:

```bash
python nn_test_for_overfitting_fast.py \
  --train-dir ../../eurosat-splits/train/images \
  --test-dir ../../eurosat-splits/test/images \
  --sample-dir ../REPA/samples/eurosat-finetune-lora-adaln-io-sit-l-2-satdino-b/SiT-L-2-0070080-size-256-vae-ema-cfg-1.0-seed-0-sde
```

- 04/11 Natasha. Made a memorization check based on encoded features (by SSCD by default: Meta's copy-detection model) rather than pixel values. It can be called as e.g.:

 ```bash
PYTHONPATH=. python ../evaluation/memorization_satdino.py \
  --train-dir ../../eurosat-splits/train/images \
  --ref eurosat_test_ref.npz \
  samples/eurosat-finetune-lora-adaln-io-sit-l-2-satdino-b/SiT-L-2-0060590-size-256-vae-ema-cfg-1.0-seed-0-sde.npz
```

- 04/11 Natasha. Made a script to compute Frechet distance using different features than evaluator.py (satdino or ssl4eo-moco instead of Inception features). It can be called as e.g.

```bash
PYTHONPATH=. python ../evaluation/fd_satdino.py \
  eurosat_test_ref.npz \
  samples/eurosat-finetune-lora-adaln-io-sit-l-2-satdino-b/SiT-L-2-0060590-size-256-vae-ema-cfg-1.0-seed-0-sde.npz \
  --encoder satdino-b
```

```bash
PYTHONPATH=. python ../evaluation/fd_satdino.py \
  eurosat_test_ref.npz \
  samples/eurosat-finetune-lora-adaln-io-sit-l-2-satdino-b/SiT-L-2-0060590-size-256-vae-ema-cfg-1.0-seed-0-sde.npz \
  --encoder ssl4eo-moco
```

  
