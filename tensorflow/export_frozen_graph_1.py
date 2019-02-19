import tensorflow as tf
from tensorflow.python.platform import gfile

# Start TensorFlow session
with tf.Session() as sess:

    # Load the graph
    metagraph = tf.saved_model.loader.load(sess, ['serve'], 'backup')

    # TensorBoard
    tf.summary.FileWriter('./logs/graph', sess.graph)

    # Export the inference graph
    graph_def = sess.graph.as_graph_def()
    with gfile.GFile('graph.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())

    # Remove useless nodes
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['myPrediction'])

    # Export the inference frozen graph
    with gfile.GFile('frozen_graph.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())