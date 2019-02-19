import numpy as np
import tensorflow as tf

from mnist import load_mnist_dataset

# Load the MNIST dataset
train_images, train_labels, test_images, test_labels = load_mnist_dataset()

with tf.Session() as sess:

    # Load the graph
    metagraph = tf.saved_model.loader.load(sess, ["serve"], 'backup')

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

