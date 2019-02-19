# Save checkpoints - summary

To save a checkpoint, one first need to define a `tf.train.Saver`. The `saver` is then used during the TensorFlow session to save the checkpoints at a given frequency.

In this example, I define the following `saver`:

```python
saver = tf.train.Saver(max_to_keep=3)
```

With the option `max_to_keep=3`, one can understand that the `saver` will only keep the last 3 checkpoints created. Other checkpoints created during the session will be deleted one by one as they are replaced by newer checkpoints. 

Please refer to the [official documentation](https://www.tensorflow.org/api_docs/python/tf/train/Saver#__init__) for the list of all available options.

Then, the `saver` can be used during the session:

```python
# Some code here
	
with tf.Session() as sess:

	# Some code here
	
	for i in range(nb_epochs):

        # Some code here

        saver.save(sess, 'checkpoints/my_model', global_step=i, write_meta_graph=True)
```

This code is a simplified version of the original code. Some notes:

* Checkpoints are saved in the folder `checkpoints` and files are prefixed by `my_model`.
* A new checkpoint is saved at each epoch.
* Using the `global_step=i` option, a number is appened to each checkpoint filename (e.g. `my_model-5`). This allows for instance not to overwrite the checkpoints written in a previsous session. In this example, we simply add the epoch number to the checkpoint filename.
* Using the `write_meta_graph` option, we define here that we want to write the graph too, and not only the weights. Since the graph (network structure) can be heavy, this allows for instance to write the graph at the first epoch, and then simply write the weights during following epochs.


# Save checkpoints - entire code

Bellow is the entire code to train the model and save the checkpoints:

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
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='myInput')
y = tf.placeholder(tf.float32, shape=[None, 10], name='myOutput')

# Model without the softmax
logits = create_model(x)

# Model with the softmax
y_pred = tf.nn.softmax(logits, name='myPrediction')

# Accuracy
acc = tf.contrib.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(y_pred,1), name='myAccuracy')

# Loss function
loss = tf.losses.softmax_cross_entropy(y, logits)

# Optimizer
optimizer = tf.train.AdamOptimizer(1e-3)

# Train
train = optimizer.minimize(loss)

# Checkpoint: define saver
saver = tf.train.Saver(max_to_keep=3)

# Parameters
nb_epochs = 5
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

        # Checkpoint: Save the model
        saver.save(sess, 'checkpoints/my_model', global_step=i, write_meta_graph=True)


```

Notice that I nammed some of the nodes in the graph (`myInput`, `myOutput`, `myPrediction` and `myAccuracy`). This will be useful when loading the checkpoint.

The output logs should look like this:

```
Training
  Epoch:  0, Loss: 0.144, Accuracy (train): 88.5, Accuracy (test): 88.6
  Epoch:  1, Loss: 0.152, Accuracy (train): 93.4, Accuracy (test): 93.7
  Epoch:  2, Loss: 0.213, Accuracy (train): 94.8, Accuracy (test): 95.1
  Epoch:  3, Loss: 0.076, Accuracy (train): 95.5, Accuracy (test): 95.5
  Epoch:  4, Loss: 0.055, Accuracy (train): 95.8, Accuracy (test): 95.6
```

After executing this code, checkpoints are saved into the `checkpoints` folder as explained earlier. The list of checkpoint files is the following:

```
checkpoint
my_model-2.data-00000-of-00001
my_model-2.index
my_model-2.meta
my_model-3.data-00000-of-00001
my_model-3.index
my_model-3.meta
my_model-4.data-00000-of-00001
my_model-4.index
my_model-4.meta
```

* `checkpoint` contains the list of the last checkpoints created.
* Only 3 checkpoints are present, as expected.
	* `*.meta` files correspond to the graph structure. This could only be created once as the structure does not change over training.
	* `*.index` and `*.data*` files are the actual model weigths.


# Load a checkpoint

Bellow is a code that loads and tests a model previously saved as a checkpoints.

```python
import numpy as np
import tensorflow as tf

from mnist import load_mnist_dataset

# Load the MNIST dataset
train_images, train_labels, test_images, test_labels = load_mnist_dataset()

with tf.Session() as sess:

    # Get the last checkpoint prefix
    checkpoint_path = tf.train.latest_checkpoint('checkpoints')

    # Load the graph structure
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta')

    # Load the weights
    saver.restore(sess, checkpoint_path)

    # Get the tensors
    x = sess.graph.get_tensor_by_name('myInput:0')
    y = sess.graph.get_tensor_by_name('myOutput:0')
    y_pred = sess.graph.get_tensor_by_name('myPrediction:0')
    acc = sess.graph.get_tensor_by_name('myAccuracy/Mean:0')

    # Testing the 10 first test images
    print '\nTesting'
    for idx in range(10):
        groundtruth = test_labels[idx]
        prediction = np.argmax(sess.run(y_pred, feed_dict={x: [test_images[idx]]})[0])
        print '  Image %d, Groundtruth = %d, Prediction = %d' % (idx, groundtruth, prediction)

    # Compute accuracies
    train_labels_one_hot = sess.run(tf.one_hot(train_labels, 10))
    test_labels_one_hot = sess.run(tf.one_hot(test_labels, 10))
    accuracy_train = acc.eval(feed_dict={x: train_images, y: train_labels_one_hot})
    accuracy_test = acc.eval(feed_dict={x: test_images, y: test_labels_one_hot})
    print '\nAccuracies'
    print '  Train: %4.1f' % (100.*accuracy_train)
    print '  Test:  %4.1f' % (100.*accuracy_test)
```

Basically, this is what is done:

* Load the MNIST dataset. See [here](mnist.md) for more information.
* Indentify the last checkpoint.
* Load the graph structure from the checkpoint.
* Load the weight from the checkpoint.
* Access important nodes in the graph thanks to nodes'name: input, output, prediction and accuracy.
* Display the groundtruth and the prediction labels for the 10 first testing images.
* Compute the train and test accuracies.

The output logs are the following:

```
Testing
  Image 0, Groundtruth = 7, Prediction = 7
  Image 1, Groundtruth = 2, Prediction = 2
  Image 2, Groundtruth = 1, Prediction = 1
  Image 3, Groundtruth = 0, Prediction = 0
  Image 4, Groundtruth = 4, Prediction = 4
  Image 5, Groundtruth = 1, Prediction = 1
  Image 6, Groundtruth = 4, Prediction = 4
  Image 7, Groundtruth = 9, Prediction = 1
  Image 8, Groundtruth = 5, Prediction = 5
  Image 9, Groundtruth = 9, Prediction = 9

Accuracies
  Train: 95.8
  Test:  95.6
```

The good news here is that we can see that the train and test accuracies are identical to the ones obtained at training time (last epoch).