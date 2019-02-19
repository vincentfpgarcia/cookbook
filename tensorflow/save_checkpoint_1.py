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
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="myInput")
y = tf.placeholder(tf.float32, shape=[None, 10], name="myOutput")

# Model without the softmax
logits = create_model(x)

# Model with the softmax
y_pred = tf.nn.softmax(logits, name="myPrediction")

# Accuracy
acc = tf.contrib.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(y_pred,1), name="myAccuracy")

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

