#!/usr/bin/env python3

import argparse
from glob import glob
from typing import Iterable
import os
from functools import reduce

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

def samples_to_tensor(samples: Iterable[Image]):
    out = np.stack(
        [
            np.array(s.convert("RGB"))
            for s in samples
        ],
        axis=0
    )
    shape = (out.shape[0], reduce(lambda x, y: x * y, out.shape[1:]))
    return out.reshape(shape)

def main(args):
    # We use ** because training samples are in nested dirs to indicate class.
    train_sample_paths = glob(os.path.join(args.train_dir, "**", "*.png"))
    train_set = samples_to_tensor([
        Image.open(p)
        for p in train_sample_paths
    ])

    output_sample_paths = glob(os.path.join(args.sample_dir, "*.png"))
    output_set = samples_to_tensor([
        Image.open(p)
        for p in output_sample_paths
    ])

    # Euclidian metric means that the metric used to evaluate distance in NN is per-pixel MSE.
    # There are other options for measuring overfitting
    nbrs = NearestNeighbors(n_neighbors=1, metric="l2").fit(train_set)
    distances, _ = nn.kneighbors(output_set)

    print(f"Per-Pixel MSE 1-NN:")
    print(f"\tAverage distance: {np.mean(distances)}")
    print(f"\tMedian distance: {np.median(distances)}")
    print(f"\t[Min, Max]: [{np.min(distances)}, {np.max(distances)}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--train-dir", type=str, required=True)

    args = parser.parse_args()
    main(args)
