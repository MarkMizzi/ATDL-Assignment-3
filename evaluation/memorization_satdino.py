"""Nearest-neighbour memorization check in a frozen encoder's feature space (SSCD by default).

Each generated image is matched to its nearest training image (cosine distance).
Held-out test images, which cannot be memorized, give the baseline distribution:
a generated image closer to the training set than the --percentile-th test image
is flagged. Without memorization the flagged fraction is about --percentile %.

Usage:
    python memorization_satdino.py --train-dir /data/eurosat-splits/train/images \
        --ref /data/eurosat_test_ref.npz samples/a.npz [samples/b.npz ...] --encoder sscd
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from eval_encoders import ENCODERS, load_encoder
from fd_satdino import extract_features, load_images

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def list_images(directory):
    paths = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(directory)
        for name in names if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not paths:
        raise ValueError(f"No images found under {directory}")
    return paths


def read_images(paths):
    return np.stack([np.array(Image.open(p).convert('RGB')) for p in paths])


def directory_features(encoder, paths, batch_size, device, chunk=1024):
    feats = []
    for start in tqdm(range(0, len(paths), chunk), desc="Training images"):
        feats.append(extract_features(encoder, read_images(paths[start:start + chunk]), batch_size, device))
    return np.concatenate(feats)


@torch.no_grad()
def nearest_neighbours(queries, keys, device, chunk=1024):
    keys = F.normalize(torch.from_numpy(keys).float().to(device), dim=1)
    queries = F.normalize(torch.from_numpy(queries).float().to(device), dim=1)
    dists, indices = [], []
    for start in range(0, len(queries), chunk):
        sims, idx = (queries[start:start + chunk] @ keys.T).max(dim=1)
        dists.append(1 - sims)
        indices.append(idx)
    return torch.cat(dists).cpu().numpy(), torch.cat(indices).cpu().numpy()


def save_pairs(path, samples, sample_idx, train_paths, train_idx, size=128):
    rows = []
    for s, t in zip(sample_idx, train_idx):
        generated = Image.fromarray(samples[s]).resize((size, size))
        nearest = Image.open(train_paths[t]).convert('RGB').resize((size, size))
        rows.append(np.concatenate([np.array(generated), np.array(nearest)], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main(args):
    device = torch.device(args.device)
    encoder = load_encoder(args.encoder, device)

    train_paths = list_images(args.train_dir)
    train_feats = directory_features(encoder, train_paths, args.batch_size, device)
    ref_feats = extract_features(encoder, load_images(args.ref), args.batch_size, device)

    ref_dists, _ = nearest_neighbours(ref_feats, train_feats, device)
    threshold = np.percentile(ref_dists, args.percentile)
    print(f"Train: {len(train_paths)} images; test baseline: {len(ref_dists)} images")
    print(f"  test->train NN cosine distance: median {np.median(ref_dists):.4f}, "
          f"p{args.percentile:g} threshold {threshold:.4f}")

    for sample_path in args.samples:
        samples = load_images(sample_path)
        dists, idx = nearest_neighbours(
            extract_features(encoder, samples, args.batch_size, device), train_feats, device,
        )
        flagged = (dists < threshold).mean() * 100
        print(f"{sample_path} ({len(samples)} images)")
        print(f"  gen->train NN cosine distance: median {np.median(dists):.4f}, min {dists.min():.4f}")
        print(f"  below test p{args.percentile:g}: {flagged:.2f}% (about {args.percentile:g}% expected without memorization)")

        closest = np.argsort(dists)[:args.num_pairs]
        pairs_path = os.path.splitext(sample_path)[0] + "_nn_pairs.png"
        save_pairs(pairs_path, samples, closest, train_paths, idx[closest])
        print(f"  closest pairs (generated | nearest training image): {pairs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", nargs="+", help="One or more generated .npz files.")
    parser.add_argument("--train-dir", required=True, help="Directory of training images (searched recursively).")
    parser.add_argument("--ref", required=True, help="Held-out real images .npz (e.g. EuroSAT test split).")
    parser.add_argument("--encoder", choices=ENCODERS, default="sscd", help="Frozen feature extractor.")
    parser.add_argument("--percentile", type=float, default=1.0, help="Test-distance percentile used as threshold.")
    parser.add_argument("--num-pairs", type=int, default=16, help="Closest generated/training pairs to save.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
