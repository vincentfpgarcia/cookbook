import tensorflow as tf
from tensorflow.python.platform import gfile

# Start TensorFlow session
with tf.Session() as sess:

    # Load the graph
    metagraph = tf.saved_model.loader.load(sess, ['serve'], 'backup')

    # Mapping values
    inputs_mapping = dict(metagraph.signature_def['serving_default'].inputs)
    outputs_mapping = dict(metagraph.signature_def['serving_default'].outputs)

    # Get the tensors
    x = sess.graph.get_tensor_by_name(inputs_mapping['input'].name)
    y_pred = sess.graph.get_tensor_by_name(outputs_mapping['prediction'].name)

    # Convert the graph from session
    converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [x], [y_pred])
    tflite_model = converter.convert()
    open('model.tflite', 'wb').write(tflite_model)