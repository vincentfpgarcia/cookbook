import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from mnist import load_mnist_dataset

# Load the MNIST dataset
train_images, train_labels, test_images, test_labels = load_mnist_dataset()

# Load the graph def
with tf.gfile.GFile('frozen_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Create a default graph
with tf.Graph().as_default() as graph:

    # Import the graph
    tf.import_graph_def(graph_def, name='prefix')
    
    # Start the session
    with tf.Session(graph=graph) as sess:

        # TensorBoard
        tf.summary.FileWriter('./logs/graph', sess.graph)

        # Get the tensors
        x = sess.graph.get_tensor_by_name('prefix/myInput:0')
        y_pred = sess.graph.get_tensor_by_name('prefix/myPrediction:0')

        # Testing the 10 first test images
        print '\nTesting'
        for idx in range(10):
            groundtruth = test_labels[idx]
            prediction = np.argmax(sess.run(y_pred, feed_dict={x: [test_images[idx]]})[0])
            print '  Image %d, Groundtruth = %d, Prediction = %d' % (idx, groundtruth, prediction)