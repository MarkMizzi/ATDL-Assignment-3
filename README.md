# ATDL Assignment 3

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

|    | Training Iterations | Training Dataset | Vision Model              | From Scratch/Pre-trained     | w. REPA? | Target Repr. (for representation alignment) | Depth | By who?    |
|----|---------------------|------------------|---------------------------|------------------------------|----------|---------------------------------------------|-------|------------|
| | **Baselines** | | | | | | | |
| 1  | 60590               | EuroSAT          | SiT-L/2                   | From Scratch                 | No       | N/A                                         | N/A   | Mark       |
| 2  | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | No       | N/A                                         | N/A   | Mark       |
| 3  | 60590               | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes      | DINOv2-b                                    | 8     | Mark       |
| 4  | 60590               | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes      | DINOv2-L                                    | 8     | Mark       |
| 5  | 60590               | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes      | DINOv2-g                                    | 8     | Mark       |
| 6  | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes      | DINOv2-b                                    | 8     | Mark       |
| 7  | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes      | DINOv2-L                                    | 8     | Mark       |
| 8  | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes      | DINOv2-g                                    | 8     | Mark       |
| | **Base Models** | | | | | | |
| 9  | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes      | SatDINO-B                                   | 8     | Natasha    |
| 10 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes      | SatDINO-S                                   | 8     | Natasha    |
| 11 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained w/ REPA DINOv2-L | Yes      | SatDINO-B + DINOv2-L                        | 8     | Natasha    |
| 12 | 60590               | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes      | SatDINO-B                                   | 8     | Natasha    |
| 13 | 60590               | EuroSAT          | SiT-L/2                   | From Scratch                 | Yes      | SatDINO-S                                   | 8     | Mark       |
| | **LoRa fine-tuning** | | | | | | |
| 9  | 60590               | EuroSAT          | SiT-L/2 (LoRa fine-tuned) | Pre-trained w/ REPA DINOv2-L | Yes      | SatDINO-S                                   | 8     | Natasha    |
| 10 | 60590               | EuroSAT          | SiT-L/2 (LoRa fine-tuned) | Pre-trained w/ REPA DINOv2-L | Yes      | SatDINO-B                                   | 8     | Natasha    |
| | **Depth ablation** | | | | | | |
| 11 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes      | SatDINO-B                                   | 6     | Frederik   |
| 12 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes      | SatDINO-B                                   | 10    | Frederik   |
| 13 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes      | SatDINO-B                                   | 12    | Frederik   |
| 14 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes      | SatDINO-B                                   | 14    | Frederik   |
| 15 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes      | SatDINO-B                                   | 16    | Frederik   |
| 16 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes      | SatDINO-B                                   | S₈ → T₉    | Frederik   |
| | **Student size ablation** | | | | | |
| 17 | 60590               | EuroSAT          | SiT-S/2                   | Pre-trained                  | Yes      | SatDINO-B                                   | 8     | Ondrej     |
| 18 | 60590               | EuroSAT          | SiT-B/2                   | Pre-trained                  | Yes      | SatDINO-B                                   | 8     | Ondrej     |
| 19 | 60590               | EuroSAT          | SiT-XL/2                  | Pre-trained                  | Yes      | SatDINO-B                                   | 8     | Ondrej     |
| | **REPA on multiple layers** | | | | | |
| 20 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer)  | SatDINO-B     | Sₐₗₗ → T₁₂     | Frederik   |
| 21 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer)  | SatDINO-B     | Sₐₗₗ → T₉     | Frederik   |
| 22 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer)  | SatDINO-B     | Sₐₗₗ → Tₐₗₗ     | Frederik   |
| 23 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer)  | SatDINO-B     | Sₐₗₗ → Tₘₑₐₙ    | Frederik   |
| 24 | 60590               | EuroSAT          | SiT-L/2                   | Pre-trained                  | Yes (Multi-layer)  | SatDINO-B     | Sₘₑₐₙ → Tₘₑₐₙ     | Frederik   |

