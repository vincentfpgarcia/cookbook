import tfcoreml
 
# Path to the input frozen graph
tf_path = 'frozen_graph.pb'

# Path to the output ML Model
coreml_path = 'frozen_graph.mlmodel'

# Input and output tensor names
input_tensor_name = 'myInput:0'
output_tensor_name = 'myPrediction:0'

# Input tensor shape
input_tensor_shapes = {input_tensor_name: [1, 28, 28, 1]}

# Output labels
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Convert the model
coreml_model = tfcoreml.convert(
        tf_model_path = tf_path,
        mlmodel_path = coreml_path,
        output_feature_names = [output_tensor_name],
        class_labels = labels,
        input_name_shape_dict = input_tensor_shapes,
        image_input_names = [input_tensor_name],
        red_bias = 0,
        green_bias = 0,
        blue_bias = 0,
        image_scale = 1.0/255.0)