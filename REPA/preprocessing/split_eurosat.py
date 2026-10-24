import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.npy'}


def read_labels(directory):
    with (directory / 'dataset.json').open(encoding='utf-8') as handle:
        entries = json.load(handle)['labels']
    if entries is None:
        raise ValueError(f"Missing labels in {directory / 'dataset.json'}")
    labels = {}
    for name, label in entries:
        name = name.replace('\\', '/')
        if name in labels:
            raise ValueError(f"Duplicate label entry: {name}")
        if type(label) is not int or not 0 <= label < 10:
            raise ValueError(f"Expected EuroSAT class ID 0-9 for {name}, got {label}")
        labels[name] = label
    return labels


def split_dataset(source, dest, seed=0, train_fraction=0.7, val_fraction=0.15,
                  mode='copy'):
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if not all(math.isfinite(value) and 0 < value < 1
               for value in (train_fraction, val_fraction)):
        raise ValueError("Train and validation fractions must be finite and between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a positive test fraction")
    if mode not in ('copy', 'hardlink'):
        raise ValueError("Mode must be copy or hardlink")
    if dest.exists():
        raise ValueError(f"Destination already exists; choose a new directory: {dest}")
    if any(dest.is_relative_to(source / folder) for folder in ('images', 'vae-sd')):
        raise ValueError("Destination must be outside the source images/ and vae-sd/ folders")

    images_dir, features_dir = source / 'images', source / 'vae-sd'
    image_names = sorted(
        path.relative_to(images_dir).as_posix()
        for path in images_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    feature_names = sorted(
        path.relative_to(features_dir).as_posix()
        for path in features_dir.rglob('*')
        if path.is_file() and path.suffix.lower() == '.npy'
    )
    if not image_names or len(image_names) != len(feature_names):
        raise ValueError("Source must contain equal, nonzero counts of images and VAE latents")
    labels = read_labels(features_dir)
    if set(labels) != set(feature_names):
        raise ValueError("Latent metadata must label every latent file exactly once")

    image_labels = None
    if (images_dir / 'dataset.json').exists():
        with (images_dir / 'dataset.json').open(encoding='utf-8') as handle:
            has_image_labels = json.load(handle).get('labels') is not None
        if has_image_labels:
            image_labels = read_labels(images_dir)
            if set(image_labels) != set(image_names):
                raise ValueError("Image metadata must label every image file exactly once")

    samples = []
    by_class = defaultdict(list)
    for image_name, feature_name in zip(image_names, feature_names):
        label = labels[feature_name]
        if image_labels is not None and image_labels[image_name] != label:
            raise ValueError(f"Image/latent label mismatch: {image_name}, {feature_name}")
        by_class[label].append(len(samples))
        samples.append({'image': image_name, 'latent': feature_name, 'label': label})
    if set(by_class) != set(range(10)):
        raise ValueError("Expected all ten EuroSAT classes with IDs 0-9")

    rng = random.Random(seed)
    split_indices = {'train': [], 'val': [], 'test': []}
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        train_count = int(len(indices) * train_fraction)
        val_count = int(len(indices) * val_fraction)
        if min(train_count, val_count, len(indices) - train_count - val_count) < 1:
            raise ValueError(f"Class {label} is too small for three nonempty splits")
        split_indices['train'].extend(indices[:train_count])
        split_indices['val'].extend(indices[train_count:train_count + val_count])
        split_indices['test'].extend(indices[train_count + val_count:])

    transfer = shutil.copy2 if mode == 'copy' else os.link
    manifest = {
        'source': str(source), 'seed': seed, 'mode': mode,
        'fractions': {
            'train': train_fraction, 'val': val_fraction,
            'test': 1 - train_fraction - val_fraction,
        },
        'splits': {},
    }
    dest.mkdir(parents=True, exist_ok=False)
    for split, indices in split_indices.items():
        selected = [samples[index] for index in sorted(indices)]
        metadata = {'images': [], 'vae-sd': []}
        for sample in selected:
            for folder, field in (('images', 'image'), ('vae-sd', 'latent')):
                name = sample[field]
                target = dest / split / folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                transfer(source / folder / name, target)
                metadata[folder].append([name, sample['label']])
        for folder, entries in metadata.items():
            with (dest / split / folder / 'dataset.json').open('w', encoding='utf-8') as handle:
                json.dump({'labels': entries}, handle, indent=2)
        manifest['splits'][split] = {
            'count': len(selected),
            'class_counts': dict(sorted(Counter(sample['label'] for sample in selected).items())),
            'samples': selected,
        }
    with (dest / 'split_manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Split preprocessed RGB EuroSAT images and latents by class without re-encoding.",
    )
    parser.add_argument('--source', type=Path, required=True,
                        help="Directory containing paired images/ and vae-sd/ folders.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="New output directory outside the source images/ and vae-sd/ folders.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--val-fraction', type=float, default=0.15)
    parser.add_argument('--mode', choices=['copy', 'hardlink'], default='copy',
                        help="Hardlinks avoid duplicate storage but require the same filesystem.")
    args = parser.parse_args()
    try:
        manifest = split_dataset(
            args.source, args.dest, args.seed, args.train_fraction,
            args.val_fraction, args.mode,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Split failed: {error}\n")
    for split, info in manifest['splits'].items():
        print(f"{split}: {info['count']} samples; class counts: {info['class_counts']}")
    print(f"Train with --data-dir={args.dest / 'train'}")


if __name__ == '__main__':
    main()