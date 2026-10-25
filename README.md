# ATDL Assignment 3

## Sharing Data and Models

Since the project required sharing data and models which are larger than what is feasible (or good practice) to store on GitHub, we used [this link here](https://sharepoint.markmizzi.com/atdl-assignment-3) as a sharepoint, which hosts the files on Mark's personal storage server. The access password for this share is `MnmDYnS@a!eDt878`.

## Artifacts

Artifacts such as trained models and images generated for evaluation are available at [this link](https://sharepoint.markmizzi.com/atdl-assignment-3/), and are organized in the following format:

```text
├── models
|   └── linear-sit-l-2-<enc-type>-enc<enc-depth>
|       ├── checkpoints                              
|       |   ├── 0400000.pt                           # Final training checkpoint, containing trained model + optimizer state
|       |   └── <other-checkpoint>.pt
|       └── samples                                  # Samples generated for evaluation
|           ├── SiT-L-2-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde.npz             # Compressed archive of evaluation samples
|           └── SiT-L-2-<other-checkpoint>-size-256-vae-ema-cfg-1.0-seed-0-sde.npz  # Same as above but for checkpoint ≠ 400K
└── VIRTUAL_imagenet256_labeled.npz                  # Reference latent for ImageNet 256x256
```

The images generated for evaluation can be extracted from the `.npz` archives using the [./extract_images_from_npz.py](./extract_images_from_npz.py), e.g.
```bash
python extract_images_from_npz.py SiT-L-2-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde.npz outdir
```

## Scope 

|    | Training Iterations | Training Dataset | Vision Model              | w. REPA? | Target Repr. (for representation alignment) | Depth | By who?    | Training time       | Energy Estimate* |
|----|---------------------|------------------|---------------------------|----------|---------------------------------------------|-------|------------|---------------------|------------------|
| | **Baselines** | | | | | | | | |
| 1  | TBD                 | EuroSAT          | SiT-L/2                   | No       | N/A                                         | N/A   | Mark       |                     |                  |
| 2  | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | DINOv2-s                                    | 8     | Mark       |                     |                  |
| 3  | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | DINOv2-b                                    | 8     | Mark       |                     |                  |
| 4  | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | DINOv2-L                                    | 8     | Mark       |                     |                  |
| 5  | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | DINOv2-g                                    | 8     | Mark       |                     |                  |
| | **Base Models** | | | | | | | | |
| 6  | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | SatDINO-S                                   | 8     |            |                     |                  |
| 7  | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | SatDINO-B                                   | 8     |            |                     |                  |
| | **LoRa fine-tuning** | | | | | | | | |
| 8  | TBD                 | EuroSAT          | SiT-L/2 (LoRa fine-tuned) | Yes      | SatDINO-S                                   | 8     |            |                     |                  |
| 9  | TBD                 | EuroSAT          | SiT-L/2 (LoRa fine-tuned) | Yes      | SatDINO-B                                   | 8     |            |                     |                  |
| | **Depth ablation** | | | | | | | | |
| 10 | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | SatDINO-B                                   | 6     |            |                     |                  |
| 11 | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | SatDINO-B                                   | 10    |            |                     |                  |
| 12 | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | SatDINO-B                                   | 12    |            |                     |                  |
| 13 | TBD                 | EuroSAT          | SiT-L/2                   | Yes      | SatDINO-B                                   | 14    |            |                     |                  |
| | **Student size ablation** | | | | | | | |
| 14 | TBD                 | EuroSAT          | SiT-S/2                   | Yes      | SatDINO-B                                   | 8     |            |                     |                  |
| 15 | TBD                 | EuroSAT          | SiT-B/2                   | Yes      | SatDINO-B                                   | 8     |            |                     |                  |
| 16 | TBD                 | EuroSAT          | SiT-XL/2                  | Yes      | SatDINO-B                                   | 8     |            |                     |                  ||
| | **REPA on multiple layers** | | | | | | |


TODO: pls fill this in last section Frederik as not sure what configs you had in mind
