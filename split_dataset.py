import numpy as np
import struct
import argparse
import os
import random


def load_images(file_path):
    with open(file_path, "rb") as f:
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic_number != 2051:
            raise ValueError(f"Invalid magic number {magic_number}, expected 2051")
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
    return images


def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic_number, num_labels = struct.unpack(">II", f.read(8))
        if magic_number != 2049:
            raise ValueError(f"Invalid magic number {magic_number}, expected 2049")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def save_images(file_path, images):
    with open(file_path, 'wb') as f:
        num_images = len(images)
        rows, cols = images.shape[1], images.shape[2]
        f.write(struct.pack(">IIII", 2051, num_images, rows, cols))
        f.write(images.tobytes())


def save_labels(file_path, labels):
    with open(file_path, 'wb') as f:
        num_labels = len(labels)
        f.write(struct.pack(">II", 2049, num_labels))
        f.write(labels.tobytes())


def split_data(images, labels, split_ratio):
    num_samples = len(images)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    split_index = int(num_samples * split_ratio)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    val_images = images[val_indices]
    val_labels = labels[val_indices]
    
    return train_images, train_labels, val_images, val_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, required=True)
    parser.add_argument('--labels-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images = load_images(args.images_path)
    labels = load_labels(args.labels_path)

    train_images, train_labels, val_images, val_labels = split_data(images, labels, args.train_ratio)

    save_images(os.path.join(args.output_dir, "train-images-idx3-ubyte"), train_images)
    save_labels(os.path.join(args.output_dir, "train-labels-idx1-ubyte"), train_labels)

    save_images(os.path.join(args.output_dir, "val-images-idx3-ubyte"), val_images)
    save_labels(os.path.join(args.output_dir, "val-labels-idx1-ubyte"), val_labels)


if __name__ == '__main__':
    main()