# Tensorflow

## Datasets

Tools to load and use datasets:

* [MNIST](tensorflow/mnist.md)

## Basic networks

* [Convolution network](tensorflow/basic_convnet.md)

## Save / load models

When training a model, it might be very useful to save this model. For instance, if the training crashes in the middle of the training, one might save a lot of time by loading the model as it was before the crash. It's also important to save the final network for the production deployment.

* [Checkpoints](tensorflow/save_checkpoint.md)
* [Saved_models](tensorflow/save_saved_model.md)

## Export models

* [Frozen graph](tensorflow/export_frozen_graph.md)
* [TFLite](tensorflow/export_tflite.md)

## Convert models

* [Frozen graph to CoreML model](tensorflow/convert_coreml.md)
