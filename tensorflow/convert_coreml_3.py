import coremltools
 
# Input ML model path
coreml_path = 'frozen_graph.mlmodel'

# Output compressed ML model path
coreml_float16_path = 'frozen_graph_float16.mlmodel'

# Load model spec from input file
model_spec = coremltools.utils.load_spec(coreml_path)

# Convert the weights to float16
model_spec_float16 = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)

# Save compressed model spec to output file
coremltools.utils.save_spec(model_spec_float16, coreml_float16_path)