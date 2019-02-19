import tfcoreml
 
# Path to the input frozen graph
tf_path = 'frozen_graph.pb'

# Path to the output ML Model
coreml_path = 'frozen_graph.mlmodel'

# Output tensor names
output_tensor_name = 'myPrediction:0'

# Convert the model
coreml_model = tfcoreml.convert(
        tf_model_path = tf_path,
        mlmodel_path = coreml_path,
        output_feature_names = [output_tensor_name])