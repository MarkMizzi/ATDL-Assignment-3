# ATDL Assignment 3

This repository contains the source code for the hand-in of Advanced Topics in Deep Learning (ATDL) Assignment 3. The repo is structured as follows:
```text
├── REPA/                              # Copy of REPA source code with our custom modifications
├── evaluation/                        # Custom evaluation scripts with our own evaluation metrics
├── guided-diffusion/                  # Submodule containing code adapted from the OpenAI diffusion evaluation suite. Used to collect some of the evaluation metrics reported.
└── guided-diffusion-requirements.txt  # Requirements file for the code in guided-diffusion/
```

The code for REPA was copied from [commit `67f7145` of the original repository](github.com/sihyun-yu/REPA/tree/67f714503e3892f993844aab088ffc5791c92613) and adapted for our experiments and trained models.

## Sharing Data and Models

Since the project required sharing data and models which are larger than what is feasible (or good practice) to store on GitHub, we used [this link here](https://sharepoint.markmizzi.com/atdl-assignment-3) as a sharepoint, which hosts the files on Mark's personal storage server. The access password for this share is `MnmDYnS@a!eDt878`.

## Artifacts

Artifacts such as trained models and images generated for evaluation are available at [this link](https://sharepoint.markmizzi.com/atdl-assignment-3/), and are organized in the following format:

```text
├── models
|   └── linear-sit-l-2-<enc-type>-enc<enc-depth>
|       ├── checkpoints                              
|       |   ├── 0060590.pt                           # Final training checkpoint, containing trained model + optimizer state
|       |   └── <other-checkpoint>.pt
|       └── samples                                  # Samples generated for evaluation
|       |   ├── SiT-L-2-0060590-size-256-vae-ema-cfg-1.0-seed-0-sde.npz             # Compressed archive of evaluation samples
|       |   └── SiT-L-2-0060590-size-256-vae-ema-cfg-1.0-seed-0-sde/                # Evaluation samples
|       └── samples-<other-checkpoint>               # Same as above but for checkpoint ≠ 60K
└── eurosat_test_ref.npz                             # Reference latent for test set sampled from EuroSAT
```

The images generated for evaluation can also be extracted from the `.npz` archives using the [./extract_images_from_npz.py](./extract_images_from_npz.py), e.g.
```bash
python extract_images_from_npz.py SiT-L-2-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde.npz outdir
```

## Models evaluated

|    | Training Iterations         | Training Dataset | Vision Model              | From Scratch/Pre-trained     | w. REPA?          | Target Repr. (for representation alignment) | Depth         | By who?  |
|----|-----------------------------|------------------|---------------------------|------------------------------|-------------------|---------------------------------------------|---------------|----------|
|    | **Baselines**               |                  |                           |                              |                   |                                             |               |          |
| 1  | 60590                       | EuroSAT          | SiT-L/2                   | From Scratch                 | No                | N/A                                         | N/A           | Mark     |
| 2  | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | No                | N/A                                         | N/A           | Mark     |
| 3  | 60590                       | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes               | DINOv2-b                                    | 8             | Mark     |
| 4  | 60590                       | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes               | DINOv2-L                                    | 8             | Mark     |
| 5  | 60590                       | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes               | DINOv2-g                                    | 8             | Mark     |
| 6  | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes               | DINOv2-b                                    | 8             | Mark     |
| 7  | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes               | DINOv2-L                                    | 8             | Mark     |
| 8  | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes               | DINOv2-g                                    | 8             | Mark     |
|    | **Base Models**             |                  |                           |                              |                   |                                             |               |
| 9  | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes               | SatDINO-B                                   | 8             | Natasha  |
| 10 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes               | SatDINO-S                                   | 8             | Natasha  |
| 11 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes               | SatDINO-B + DINOv2-L                        | 8             | Natasha  |
| 12 | 60590                       | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes               | SatDINO-B                                   | 8             | Natasha  |
| 13 | 60590                       | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes               | SatDINO-S                                   | 8             | Mark     |
|    | **LoRa fine-tuning**        |                  |                           |                              |                   |                                             |               |
| 9  | 60590                       | EuroSAT          | SiT-L/2 (LoRa fine-tuned) | Pre-trained w/ REPA DINOv2-L | Yes               | SatDINO-S                                   | 8             | Natasha  |
| 10 | 60590                       | EuroSAT          | SiT-L/2 (LoRa fine-tuned) | Pre-trained w/ REPA DINOv2-L | Yes               | SatDINO-B                                   | 8             | Natasha  |
|    | **Depth ablation**          |                  |                           |                              |                   |                                             |               |
| 11 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes               | SatDINO-B                                   | 6             | Frederik |
| 12 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes               | SatDINO-B                                   | 10            | Frederik |
| 13 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes               | SatDINO-B                                   | 12            | Frederik |
| 14 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes               | SatDINO-B                                   | 14            | Frederik |
| 15 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes               | SatDINO-B                                   | 16            | Frederik |
| 16 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes               | SatDINO-B                                   | S₈ → T₉       | Frederik |
|    | **Student size ablation**   |                  |                           |                              |                   |                                             |
| 17 | 60590                       | EuroSAT          | SiT-S/2                   | Pre-trained                  | Yes               | SatDINO-B                                   | 8             | Ondrej   |
| 18 | 60590                       | EuroSAT          | SiT-B/2                   | Pre-trained                  | Yes               | SatDINO-B                                   | 8             | Ondrej   |
| 19 | 60590                       | EuroSAT          | SiT-XL/2                  | Pre-trained                  | Yes               | SatDINO-B                                   | 8             | Ondrej   |
|    | **REPA on multiple layers** |                  |                           |                              |                   |                                             |
| 20 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer) | SatDINO-B                                   | Sₐₗₗ → T₁₂    | Frederik |
| 21 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer) | SatDINO-B                                   | Sₐₗₗ → T₉     | Frederik |
| 22 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer) | SatDINO-B                                   | Sₐₗₗ → Tₐₗₗ   | Frederik |
| 23 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer) | SatDINO-B                                   | Sₐₗₗ → Tₘₑₐₙ  | Frederik |
| 24 | 60590                       | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer) | SatDINO-B                                   | Sₘₑₐₙ → Tₘₑₐₙ | Frederik |

## Training

The training process is similar to the original REPA source code. You will need to set up and activate a Python virtual environment with all the dependencies in [`REPA/requirements.txt`](./REPA/requirements.txt).

### Using DINOv2 as target representation

REPA training with a DINOv2 model as the target representation can be achieved using a command like
```
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
  --exp-name=eurosat-dino-sit-l-2 \
  --learning-rate=1e-5 \
  --data-dir=/work/data/eurosat-splits/train \
  --val-dir=/work/data/eurosat-splits/val \
  --checkpointing-steps=730
```

In this case the chosen target representation is DINOv2-L (set in `--enc-type`), the model being trained has a SiT-L/2 architecture (set in `--model`) and the training/validation dirs are in `/work/data/eurosat-splits/{train,val}` (set in `--data-dir` and `--val-dir` respectively).

### Using SatDINO as target representation

REPA training with a SatDINO model as the target representation can be achieved using a command similar to the one shown above, but setting `--enc-type=satdino-vit-b` or `--enc-type=satdino-vit-s`.

### Using SatDINO+DINO as target representation

REPA training with SatDINO and a DINOv2 model simultaneously used as the target representation can be achieved using a command similar to the one shown above, but setting `--enc-type=dinov2-vit-l,satdino-vit-b` (or similar).

### Training without REPA

Training without REPA can be achieved using a command similar to the one shown above, but setting `--enc-type="none"`.

### Fine tuning

To fine tune a pre-trained checkpoint, any of the commands above can be used with the additional flag `--pretrained-path=<path-to-checkpoint>` where `<path-to-checkpoint>` is a path to the pre-trained checkpoint produced using REPA training scripts (checkpoints produced by the original REPA code are compatible).

## Evaluation
