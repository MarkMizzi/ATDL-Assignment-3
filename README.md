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
