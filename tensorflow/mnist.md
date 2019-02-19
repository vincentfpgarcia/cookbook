# MNIST dataset

The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.


# Download

The dataset can be found [on this page](http://yann.lecun.com/exdb/mnist/). Download the following four files:

* `train-images-idx3-ubyte.gz`:  training set images (9912422 bytes)
* `train-labels-idx1-ubyte.gz`:  training set labels (28881 bytes)
* `t10k-images-idx3-ubyte.gz`:   test set images (1648877 bytes)
* `t10k-labels-idx1-ubyte.gz`:   test set labels (4542 bytes) 


# Load the dataset

Bellow is a Python function that will allow to read the MNIST dataset from the four files.

```python
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
```

This code is inspired by [this code.](https://gist.github.com/akesling/5358964)

To use this function, place the script aside the dataset, and use the follwing code:

```python
from mnist import load_mnist_dataset

train_images, train_labels, test_images, test_labels = load_mnist_dataset()

print 'Train images :', train_images.shape
print 'Train labels :', train_labels.shape
print 'Test images  :', test_images.shape
print 'Test labels  :', test_labels.shape
```

By executing this code, you should get the following output:

```
Train images : (60000, 28, 28, 1)
Train labels : (60000,)
Test images  : (10000, 28, 28, 1)
Test labels  : (10000,)
```