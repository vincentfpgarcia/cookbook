import struct
import numpy as np

def load_mnist_dataset():

    # Data paths
    train_images_path = 'train-images-idx3-ubyte'
    train_labels_path = 'train-labels-idx1-ubyte'
    test_images_path = 't10k-images-idx3-ubyte'
    test_labels_path = 't10k-labels-idx1-ubyte'

    # Load train labels
    with open(train_labels_path, 'rb') as file:
        magic, num = struct.unpack(">II", file.read(8))
        train_labels = np.fromfile(file, dtype=np.int8)

    # Load train images
    with open(train_images_path, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        train_images = np.fromfile(file, dtype=np.uint8).reshape(len(train_labels), rows, cols, 1)

    # Load test labels
    with open(test_labels_path, 'rb') as file:
        magic, num = struct.unpack(">II", file.read(8))
        test_labels = np.fromfile(file, dtype=np.int8)

    # Load test images
    with open(test_images_path, 'rb') as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        test_images = np.fromfile(file, dtype=np.uint8).reshape(len(test_labels), rows, cols, 1)

    return train_images, train_labels, test_images, test_labels