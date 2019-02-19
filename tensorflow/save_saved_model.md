# Save using saved_model - summary

Let's give an example. Let's assume that we have a model graph already trained for the task of classification. This graph is made of many tensors, but 4 are of a particular interest here:

* `x`: input tensor (placeholder)
* `y`: expected output (placeholder)
* `y_pred`: output predicted
* `acc`: accuracy computed from `y` and `y_pred`

During the session, the model can be saved using the following code:

```python
tf.saved_model.simple_save(sess, 'backup', inputs={'input': x}, outputs={'output': y, 'prediction': y_pred, 'accuracy': acc})
```

Some notes:

* The model is saved on the folder `backup`.
* `input` and `outputs` maps the selected tensors (`x`, `y`, `y_pred`, and `acc`) to a chosen name. This mapping will be used when the model will be loaded (see section bellow).


# Save checkpoints - entire code

Bellow is the entire code to train the model and save using saved_model:

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

    # Save the model
    tf.saved_model.simple_save(sess, 'backup', inputs={'input': x}, outputs={'output': y, 'prediction': y_pred, 'accuracy': acc})

```

The output logs should look like this:

```
Training
  Epoch:  0, Loss: 0.069, Accuracy (train): 91.2, Accuracy (test): 91.2
  Epoch:  1, Loss: 0.053, Accuracy (train): 94.0, Accuracy (test): 94.2
  Epoch:  2, Loss: 0.032, Accuracy (train): 95.1, Accuracy (test): 95.1
  Epoch:  3, Loss: 0.039, Accuracy (train): 95.7, Accuracy (test): 95.6
  Epoch:  4, Loss: 0.017, Accuracy (train): 96.0, Accuracy (test): 95.9
```

After executing this code, the model is saved into the `backup` folder as explained earlier.


# Load a checkpoint

Bellow is a code that loads and tests a model previously saved as a checkpoints.

```python
import numpy as np
import tensorflow as tf

from mnist import load_mnist_dataset

# Load the MNIST dataset
train_images, train_labels, test_images, test_labels = load_mnist_dataset()

with tf.Session() as sess:

    # Load the graph
    metagraph = tf.saved_model.loader.load(sess, ['serve'], 'backup')

    # Mapping values
    inputs_mapping = dict(metagraph.signature_def['serving_default'].inputs)
    outputs_mapping = dict(metagraph.signature_def['serving_default'].outputs)

    # Get the tensors
    x = sess.graph.get_tensor_by_name(inputs_mapping['input'].name)
    y = sess.graph.get_tensor_by_name(outputs_mapping['output'].name)
    y_pred = sess.graph.get_tensor_by_name(outputs_mapping['prediction'].name)
    acc = sess.graph.get_tensor_by_name(outputs_mapping['accuracy'].name)

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
* Identify the last checkpoint.
* Load the graph structure from the `backup` folder.
* Get the input/output mappings from tensor names to actual tensor nodes.
* Access important nodes in the graph using these mappings: input, output, prediction and accuracy.
* Display the groundtruth and the prediction labels for the 10 first testing images.
* Compute the train and test accuracies.

Using the input and output mappings, it becomes very easy to access the interesting nodes in the graph. We could also access these nodes with their names in the graph though, but we should for instance use TensorBoard to visualize these nodes in the graph, get their name, then load the nodes in TensorFlow.

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
  Image 7, Groundtruth = 9, Prediction = 9
  Image 8, Groundtruth = 5, Prediction = 5
  Image 9, Groundtruth = 9, Prediction = 9

Accuracies
  Train: 96.0
  Test:  95.9
```

The good news here is that we can see that the train and test accuracies are identical to the ones obtained at training time (last epoch).