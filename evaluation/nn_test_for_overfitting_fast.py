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
    best = torch.empty(len(queries), dtype=torch.float64)
    best_idx = torch.empty(len(queries), dtype=torch.long)
    for q0 in range(0, len(queries), chunk):
        # float64: cdist's float32 rounding can reorder or zero the smallest distances.
        q = queries[q0:q0 + chunk].to(device).double() / 255.0
        q_best = torch.full((len(q),), float("inf"), dtype=torch.float64, device=device)
        q_idx = torch.zeros(len(q), dtype=torch.long, device=device)
        for k0 in range(0, len(keys), chunk):
            k = keys[k0:k0 + chunk].double() / 255.0
            d, i = torch.cdist(q, k).min(dim=1)
            closer = d < q_best
            q_best = torch.where(closer, d, q_best)
            q_idx = torch.where(closer, i + k0, q_idx)
        best[q0:q0 + chunk] = q_best.cpu()
        best_idx[q0:q0 + chunk] = q_idx.cpu()
    return best.numpy(), best_idx.numpy()


def print_stats(name, distances, dim, resolution):
    print(f"Generated -> {name}: pixel L2 1-NN distance ([0,1] pixels, {resolution}):")
    print(f"\tAverage distance: {np.mean(distances)}")
    print(f"\tMedian distance: {np.median(distances)}")
    print(f"\t[Min, Max]: [{np.min(distances)}, {np.max(distances)}]")

    # RMSE = L2 / sqrt(H*W*3), comparable across resolutions.
    rmse = distances / np.sqrt(dim)
    print(f"Generated -> {name}: per-pixel RMSE ([0,1] pixels, resolution-independent):")
    print(f"\tAverage RMSE: {np.mean(rmse)}")
    print(f"\tMedian RMSE: {np.median(rmse)}")
    print(f"\t[Min, Max]: [{np.min(rmse)}, {np.max(rmse)}]")


def save_pair(generated_path, real_path, out_path):
    generated = Image.open(generated_path).convert("RGB")
    real = Image.open(real_path).convert("RGB").resize(generated.size, Image.NEAREST)
    pair = Image.new("RGB", (generated.width * 2, generated.height))
    pair.paste(generated, (0, 0))
    pair.paste(real, (generated.width, 0))
    pair.save(out_path)


def main(args):
    output_sample_paths = glob(os.path.join(args.sample_dir, "*.png"))
    output_set = load_images(output_sample_paths, args.resize, args.workers)
    resolution = f"{args.resize}x{args.resize}" if args.resize else "native resolution"
    device = torch.device(args.device)
    sample_dir = os.path.normpath(args.sample_dir)
    sample_parent, sample_name = os.path.dirname(sample_dir), os.path.basename(sample_dir)

    for name, real_dir in (("train", args.train_dir), ("test", args.test_dir)):
        # We use ** because real images are in nested dirs to indicate class.
        real_paths = glob(os.path.join(real_dir, "**", "*.png"))
        real_set = load_images(real_paths, args.resize, args.workers)
        distances, nearest = nearest_distances(output_set, real_set, device)
        print_stats(name, distances, output_set.shape[1], resolution)

        closest = int(np.argmin(distances))
        generated_path = output_sample_paths[closest]
        real_path = real_paths[nearest[closest]]
        pair_path = os.path.join(sample_parent, f"{sample_name}_closest_pair_{name}.png")
        save_pair(generated_path, real_path, pair_path)
        print(f"Closest pair (generated | {name}), L2 distance {distances[closest]}:")
        print(f"\t{generated_path}\n\t{real_path}\n\tSaved: {pair_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--resize", type=int, default=None,
                        help="Downscale images to this square size first, e.g. 64 (EuroSAT's native resolution).")
    parser.add_argument("--workers", type=int, default=8, help="Threads for image loading.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)
