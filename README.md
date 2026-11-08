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

## Pre-processing

This work uses the EuroSAT dataset which can be downloaded from Kaggle using the following Python script:
```python
import kagglehub 
kagglehub.login()
path = kagglehub.dataset_download("apollo2506/eurosat-dataset", output_dir="data")
```

The dataset was preprocessed using the following command:
```bash
python REPA/preprocessing/dataset_tools.py convert --source EuroSAT-data/EuroSAT/ --dest images --resolution 256x256 --transform=center-crop-dhariwal
```
and Stability variational auto-encoder (VAE) latents can be generated from the preprocessed dataset as follows:
```bash
python REPA/preprocessing/dataset_tools.py encode --source=images --dest=vae-sd
```

Train/test/validation splits of 70%/15%/15% can be generated using the following command:
```bash
python preprocessing/split_eurosat.py --source=/work/data --dest=/work/data/eurosat-splits
```
where `/work/data` can be replaced by the directory containing the downloaded EuroSAT dataset.

Note that you will need to set up and activate a Python virtual environment with all the dependencies in [`REPA/requirements.txt`](./REPA/requirements.txt) to run any of the commands above successfully.

## Training

The training process is similar to the original REPA source code. You will need to set up and activate a Python virtual environment with all the dependencies in [`REPA/requirements.txt`](./REPA/requirements.txt).

### Using DINOv2 as target representation

REPA training with a DINOv2 model as the target representation can be achieved using a command like
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

### Aligning with a hidden layer from the "teacher encoder"

To train using REPA with a hidden layer from the "teacher encoder", use a command like:
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
  --learning-rate=1e-5 \
  --data-dir=/work/data/eurosat-splits/train \
  --val-dir=/work/data/eurosat-splits/val \
  --checkpointing-steps=730
```

### Multi-layer REPA

To train using a multi-layer REPA configuration, use a command like:
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
  --learning-rate=1e-5 \
  --data-dir=/work/data/eurosat-splits/train \
  --val-dir=/work/data/eurosat-splits/val \
  --checkpointing-steps=730
```

Note that `--repa-config` references one of the configurations described in the assignment report, with these configurations being hard-coded into the training script.

### Fine tuning

To fine tune a pre-trained checkpoint, any of the commands above can be used with the additional flag `--pretrained-path=<path-to-checkpoint>` where `<path-to-checkpoint>` is a path to the pre-trained checkpoint produced using REPA training scripts (checkpoints produced by the original REPA code are compatible).

## Evaluation

During evaluation, 4000 samples generated from trained models are compared to a compressed archive (called `eurosat_test_ref.npz` in the commands below) of 15% of the images from EuroSAT which were set aside prior to training for this purpose. The archive can be obtained [here](https://sharepoint.markmizzi.com/atdl-assignment-3/eurosat_test_ref.npz).

### Generating Samples

Samples can be generated from a trained checkpoint using a command like 
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

Note in particular the options `--num-fid-samples`, `--ckpt` and `--sample-dir`, which set the number of samples generated, the checkpoint used for generating the samples, and the output directory respectively. The option `--projector-embed-dims` must also match the output dimensions of the encoder used during training. If REPA was not used during training, you must set `--projector-embed-dims=''`.

 Note also that you will need to set up and activate a Python virtual environment with all the dependencies in [`REPA/requirements.txt`](./REPA/requirements.txt).

### ADM Evaluation Suite

To collect metrics from the ADM evaluation suite for a trained model, you first need to set up and activate a Python virtual environment with all the dependencies in [`guided-diffusion-requirements.txt`](./guided-diffusion-requirements.txt).

You can then run a command like
```bash
python guided-diffusion/evaluations/evaluator.py \
    ./eurosat_test_ref.npz                         \
    <samples-dir>/SiT-L-2-0060590-size-256-vae-ema-cfg-1.0-seed-0-sde.npz
```
to collect the metrics for samples in `<samples-dir>`

### Custom Evaluation Suite

Under [./evaluation](./evaluation) is a custom suite of evaluation scripts which we have developed for this project.

One of them measures statistics on nearest neighbour distance between the training set and generated images from a model to ensure that the model is not memorizing samples. The metric used in NN is per-pixel mean square error (MSE). The script can be run using a command like
```bash
python nn_test_for_overfitting_fast.py \
  --train-dir ../../eurosat-splits/train/images \
  --test-dir ../../eurosat-splits/test/images \
  --sample-dir ../REPA/samples/eurosat-finetune-lora-adaln-io-sit-l-2-satdino-b/SiT-L-2-0070080-size-256-vae-ema-cfg-1.0-seed-0-sde
```
where `--train-dir`, `--test-dir` and `--sample-dir` should point to the directories containing the training, test and generated samples respectively.

Another of the scripts can compute Frechet distance using features from encoders other than Inception (e.g. `satdino` or `ssl4eo-moco`. It can be run using a command like
```bash
PYTHONPATH=. python ../evaluation/fd_satdino.py \
  eurosat_test_ref.npz \
  samples/eurosat-finetune-lora-adaln-io-sit-l-2-satdino-b/SiT-L-2-0060590-size-256-vae-ema-cfg-1.0-seed-0-sde.npz \
  --encoder satdino-b
```
where `--encoder` is the encoder whose features are used to compute Frechet distance, and can be set to either `satdino-b` or `ssl4eo-moco`. Note that the script must be run from the `REPA` subdirectory as it uses the `utils.py` file in that directory.

The last of our scripts computes another memorization check based on encoded features (from SSCD by default; this is Meta's copy-detection model), and can be run using a command like
```bash
PYTHONPATH=. python ../evaluation/memorization_satdino.py \
 --train-dir ../../eurosat-splits/train/images \
 --ref eurosat_test_ref.npz \
 samples/eurosat-finetune-lora-adaln-io-sit-l-2-satdino-b/SiT-L-2-0060590-size-256-vae-ema-cfg-1.0-seed-0-sde.npz
```
Note that once again the script must be run from the `REPA` subdirectory as it uses the `utils.py` file in that directory.
