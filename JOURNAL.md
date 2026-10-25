# Journal

## Dataset Preparation

- 22/10 Mark. Downloaded the EuroSAT dataset with:
  ```python
  import kagglehub 
  kagglehub.login()
  path = kagglehub.dataset_download("apollo2506/eurosat-dataset", output_dir="data")
  ```

- 23/10 Mark. Converted EuroSAT dataset to format used by REPA scripts using
  ```python
  python REPA/preprocessing/dataset_tools.py convert --source EuroSAT-data/EuroSAT/ --dest images --resolution 256x256 --transform=center-crop-dhariwal
  ```

- 23/10 Mark. Generated EuroSAT latents on UCloud server using
  ```python
  python REPA/preprocessing/dataset_tools.py encode --source=images --dest=vae-sd
  ```
  Version of REPA code used is latest of personal fork.

- 24/10 Natasha. Implemented the split of EuroSAT dataset into train/validation/test in the proportion 70/15/15. The script for splitting the data has been called as
  ```python
  python preprocessing/split_eurosat.py --source=/work/data --dest=/work/data/eurosat-splits
  ```
## Training

- 24/10 Natasha. Changed the training script to be able to choose which alignment encoder to use (also none or both DINO and SatDINO are possible). The loss(es) are computed now separately for train
    and validation datasets, and the generated example pictures are saved to WandB more often. The plan is to do the following 4 training schemes (note that --reset-projectors is ONLY used when switching from one alignment encoder to another one!):

  finetune with DINO:
  ```python
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

  ```python
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

  ```python
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

  ```python
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
    
