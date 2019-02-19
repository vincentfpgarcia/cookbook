# Export TFLite model

To illustrate how to export a TFLite model, we assume that we have already trained a model and saved it using `saved_model`. Please check [here](save_saved_model.md) to see how to do that. This allows me to simplify the code.

Let's give an example with the following code:

```python
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
```

What's done:

* Load the graph structure from the `backup` folder.
* Get the input/output mappings from tensor names to actual tensor nodes.
* Access the input `x` and prediction `y_pred` nodes.
* Export the graph as a TFLite file `model.tflite`.

Some notes:

* The input and output tensors (`x` and `y_pred` respo.) are passed as input values to the converter function.
* The TFLite file is very lite compared to the loaded model.
