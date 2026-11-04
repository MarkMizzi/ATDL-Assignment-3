#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor
from glob import glob
import os

import numpy as np
import torch
from PIL import Image


def load_images(paths, size, workers):
    def load(path):
        img = Image.open(path).convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.BOX)
        return np.asarray(img)

    with ThreadPoolExecutor(workers) as pool:
        images = list(pool.map(load, paths))
    # Kept as uint8 to save memory; scaled to [0, 1] on the device.
    return torch.from_numpy(np.stack(images)).reshape(len(images), -1)


@torch.no_grad()
def nearest_distances(queries, keys, device, chunk=1024):
    keys = keys.to(device)
    best = torch.empty(len(queries))
    for q0 in range(0, len(queries), chunk):
        q = queries[q0:q0 + chunk].to(device).float() / 255.0
        q_best = torch.full((len(q),), float("inf"), device=device)
        for k0 in range(0, len(keys), chunk):
            k = keys[k0:k0 + chunk].float() / 255.0
            q_best = torch.minimum(q_best, torch.cdist(q, k).min(dim=1).values)
        best[q0:q0 + chunk] = q_best.cpu()
    return best.numpy()


def main(args):
    # We use ** because training samples are in nested dirs to indicate class.
    train_sample_paths = glob(os.path.join(args.train_dir, "**", "*.png"))
    output_sample_paths = glob(os.path.join(args.sample_dir, "*.png"))
    train_set = load_images(train_sample_paths, args.resize, args.workers)
    output_set = load_images(output_sample_paths, args.resize, args.workers)

    distances = nearest_distances(output_set, train_set, torch.device(args.device))

    resolution = f"{args.resize}x{args.resize}" if args.resize else "native resolution"
    print(f"Pixel L2 1-NN distance ([0,1] pixels, {resolution}):")
    print(f"\tAverage distance: {np.mean(distances)}")
    print(f"\tMedian distance: {np.median(distances)}")
    print(f"\t[Min, Max]: [{np.min(distances)}, {np.max(distances)}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--resize", type=int, default=None,
                        help="Downscale images to this square size first, e.g. 64 (EuroSAT's native resolution).")
    parser.add_argument("--workers", type=int, default=8, help="Threads for image loading.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)
