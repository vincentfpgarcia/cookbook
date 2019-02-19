# Training a basic convnet

You'll find bellow a TensorFlow script to train a convnet (_convolutional neural network_) on the MNIST dataset. If you don't know how to access this dataset, please read [this file](mnist.md).

```python
import numpy as np
import tensorflow as tf

from mnist import load_mnist_dataset


def create_model(x):

    # Convolution
    conv1 = tf.layers.conv2d(x, filters=8, kernel_size=(3,3), strides=1, padding='same', activation='relu')
    conv1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # Convolution
    conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=(3,3), strides=1, padding='same', activation='relu')
    conv2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Convolution
    conv3 = tf.layers.conv2d(conv2, filters=4, kernel_size=(1,1), strides=1, padding='same', activation='relu')
    conv3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)

    # Flatten
    flat = tf.layers.flatten(conv3)

    # Fully connected layers (no softmax)
    fc = tf.layers.dense(flat, units=32, activation='relu')
    output = tf.layers.dense(fc, units=10)

    return output


# Load the MNIST dataset
train_images, train_labels, test_images, test_labels = load_mnist_dataset()

# Placeholders for the input / output tensors
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Model without the softmax
logits = create_model(x)

# Model with the softmax
y_pred = tf.nn.softmax(logits)

# Accuracy
acc = tf.contrib.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(y_pred,1), name='myAccuracy')

# Loss function
loss = tf.losses.softmax_cross_entropy(y, logits)

# Optimizer
optimizer = tf.train.AdamOptimizer(1e-3)

# Train
train = optimizer.minimize(loss)

# Parameters
nb_epochs = 10
batch_size = 32     
batch_nb = len(train_labels) / batch_size

# Start TensorFlow session
with tf.Session() as sess:

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Convert labels to one hot
    train_labels_one_hot = sess.run(tf.one_hot(train_labels, 10))
    test_labels_one_hot = sess.run(tf.one_hot(test_labels, 10))

    # Iterate through epochs
    print 'Training'
    for i in range(nb_epochs):

        # Batch
        for j in range( batch_nb ):

            # Create mini-batch (no shuffle)
            start = j * batch_size
            end = start + batch_size
            _train_images = train_images[start:end]
            _train_labels = train_labels_one_hot[start:end]

            # Train the model
            _, loss_value = sess.run([train, loss], feed_dict={x: _train_images, y: _train_labels})

        # Compute accuracies
        accuracy_train = acc.eval(feed_dict={x: train_images, y: train_labels_one_hot})
        accuracy_test = acc.eval(feed_dict={x: test_images, y: test_labels_one_hot})

        # Print
        print '  Epoch: %2d, Loss: %5.3f, Accuracy (train): %4.1f, Accuracy (test): %4.1f' % (i, loss_value, 100*accuracy_train, 100*accuracy_test)

    # Testing the 10 first test images
    print '\nTesting'
    for idx in range(10):
        groundtruth = test_labels[idx]
        prediction = np.argmax(sess.run(y_pred, feed_dict={x: [test_images[idx]]})[0])
        print '  Image %d, Groundtruth = %d, Prediction = %d' % (idx, groundtruth, prediction)

```

The code should be pretty much self explanatory. Let's give some details about specific parts.

* The function `create_model` create a convnet, layer by layer:
	* `x` is the input layer.
	* 8 3x3 convolution filters + bias + ReLU activation + max pooling
	* 16 3x3 convolution filters + bias + ReLU activation + max pooling
	* 4 1x1 convolution filters + bias + ReLU activation + max pooling
	* Fully connected layer with 32 outputs + ReLU activation
	* Fully connected layer with 10 outputs
* Notice that the last layer of the model does not have a softmax activation.
* The MNIST dataset is loaded using the `load_mnist_dataset`. See [here]
(mnist.md) for more details.
* `x` and `y` are respectively the input and expected output (groundtruth)  of the network. We use here placeholders.
* `y_pred` is the prediction of the network. It is obtained by addind a softmax activation to the created model.
* We add a node in the grah to compute the accuracy of the model (training or testing accuracy depending on which couple images/labels we plug to this node).
* Define loss and optimizer for training.
* Training and testing labels are converted into onehot labels.
* The training is performed through 10 epochs using mini batches of 32 images.
* At the end of each epoch, training and testing accuracies are computed and displayed.
* Finally, we display the groundtruth and the prediction labels for the 10 first testing images.

By executing the previous script, we got the following results. Note that one will probably have different (but similar) results when training the model on his/her computer.

```
Training
  Epoch:  0, Loss: 0.336, Accuracy (train): 88.6, Accuracy (test): 88.7
  Epoch:  1, Loss: 0.255, Accuracy (train): 93.7, Accuracy (test): 93.9
  Epoch:  2, Loss: 0.044, Accuracy (train): 95.4, Accuracy (test): 95.3
  Epoch:  3, Loss: 0.028, Accuracy (train): 95.2, Accuracy (test): 95.4
  Epoch:  4, Loss: 0.013, Accuracy (train): 96.3, Accuracy (test): 96.3
  Epoch:  5, Loss: 0.008, Accuracy (train): 96.6, Accuracy (test): 96.5
  Epoch:  6, Loss: 0.048, Accuracy (train): 96.5, Accuracy (test): 96.1
  Epoch:  7, Loss: 0.019, Accuracy (train): 96.5, Accuracy (test): 96.3
  Epoch:  8, Loss: 0.056, Accuracy (train): 97.2, Accuracy (test): 97.0
  Epoch:  9, Loss: 0.045, Accuracy (train): 97.4, Accuracy (test): 96.9

Testing
  Image 0, Groundtruth = 7, Prediction = 7
  Image 1, Groundtruth = 2, Prediction = 2
  Image 2, Groundtruth = 1, Prediction = 1
  Image 3, Groundtruth = 0, Prediction = 0
  Image 4, Groundtruth = 4, Prediction = 4
  Image 5, Groundtruth = 1, Prediction = 1
  Image 6, Groundtruth = 4, Prediction = 4
  Image 7, Groundtruth = 9, Prediction = 9
  Image 8, Groundtruth = 5, Prediction = 5
  Image 9, Groundtruth = 9, Prediction = 9
  ```