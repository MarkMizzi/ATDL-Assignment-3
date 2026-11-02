"""Fréchet distance, precision and recall between real and generated images in SatDINO CLS-feature space.

Precision/recall follow Kynkäänniemi et al. (2019) with k=3, as in the ADM evaluator.

Usage:
    python fd_satdino.py eurosat_test_ref.npz samples/a.npz [samples/b.npz ...] --model b
"""
import argparse

import numpy as np
import torch
from scipy import linalg
from torchvision.transforms import Normalize
from tqdm import tqdm

from utils import SATDINO_MODEL_IDS, SatDINOEncoder

# fMoW RGB statistics, identical to train.py's SatDINO preprocessing.
SATDINO_DEFAULT_MEAN = (0.433, 0.436, 0.407)
SATDINO_DEFAULT_STD = (0.198, 0.193, 0.195)


def load_images(path):
    images = np.load(path)['arr_0']
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8 array of shape (N, H, W, 3), got {images.dtype} {images.shape}")
    return images


@torch.no_grad()
def extract_features(encoder, images, batch_size, device):
    normalize = Normalize(SATDINO_DEFAULT_MEAN, SATDINO_DEFAULT_STD)
    features = []
    for start in tqdm(range(0, len(images), batch_size), leave=False):
        batch = torch.from_numpy(images[start:start + batch_size]).to(device)
        batch = normalize(batch.permute(0, 3, 1, 2).float() / 255.)
        features.append(encoder.model(batch).double().cpu())  # CLS token
    return torch.cat(features).numpy()


def frechet_distance(feats1, feats2, eps=1e-6):
    mu1, mu2 = feats1.mean(0), feats2.mean(0)
    sigma1, sigma2 = np.cov(feats1, rowvar=False), np.cov(feats2, rowvar=False)
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


@torch.no_grad()
def manifold_radii(feats, k, chunk=1024):
    radii = []
    for start in range(0, len(feats), chunk):
        dists = torch.cdist(feats[start:start + chunk], feats)
        radii.append(dists.kthvalue(k + 1, dim=1).values)  # k+1 skips the point itself
    return torch.cat(radii)


@torch.no_grad()
def manifold_coverage(points, centers, radii, chunk=1024):
    inside = []
    for start in range(0, len(points), chunk):
        dists = torch.cdist(points[start:start + chunk], centers)
        inside.append((dists <= radii[None]).any(dim=1))
    return torch.cat(inside).double().mean().item()


def precision_recall(ref_feats, sample_feats, k, device):
    ref = torch.from_numpy(ref_feats).float().to(device)
    sample = torch.from_numpy(sample_feats).float().to(device)
    precision = manifold_coverage(sample, ref, manifold_radii(ref, k))
    recall = manifold_coverage(ref, sample, manifold_radii(sample, k))
    return precision, recall


def main(args):
    device = torch.device(args.device)
    encoder = SatDINOEncoder(SATDINO_MODEL_IDS[args.model]).to(device)

    ref_feats = extract_features(encoder, load_images(args.ref), args.batch_size, device)
    print(f"Reference: {args.ref} ({len(ref_feats)} images)")
    for sample_path in args.samples:
        sample_feats = extract_features(encoder, load_images(sample_path), args.batch_size, device)
        fd = frechet_distance(ref_feats, sample_feats)
        precision, recall = precision_recall(ref_feats, sample_feats, args.k, device)
        name = f"SatDINO-{args.model.upper()}"
        print(f"{sample_path} ({len(sample_feats)} images)")
        print(f"  FD-{name}: {fd:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref", help="Reference .npz of real images (arr_0, uint8 NHWC).")
    parser.add_argument("samples", nargs="+", help="One or more generated .npz files.")
    parser.add_argument("--model", choices=sorted(SATDINO_MODEL_IDS), default="b", help="SatDINO ViT size.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k", type=int, default=3, help="Nearest-neighbour count for precision/recall manifolds.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
