# PyTorch to CoreML conversion

## Forewords

The model I'll use for the following documentation is [FFDNet](http://www.ipol.im/pub/art/2019/231/). FFDNet is mainly a convnet used for image denoising. This model takes 2 inputs (an RGB image and the noise variance value) and outputs the denoised image. Note that in the original code, the inference returned the noise map. I've simply subtracted the noise map from the input image to directly return the denoised image. No big deal.

The PyTorch model inputs and outputs images with values in [0,1] encoded on float32. Since most of images are stored on uint8, we will need to convert the image values.

I will use the following Python packages. Most of them were installed through Anaconda, and some of them had to be installed using Pip as they were not available in Anaconda. Here is the list:

- torch
- coremltools
- onnx_coreml
- pillow
- numpy


## PyTorch to ONNX conversion

As of today, the most straightful way to convert a PyTorch model into a CoreML model is through [ONNX](https://onnx.ai/). Here is an example:

```python
import torch
from model import FFDNet

# Load the PyTorch model
state_dict = torch.load('FFDNet.pth', map_location='cpu')
model = FFDNet()
model.load_state_dict(state_dict)

# Initialize weights (only necessary for my model)
model.init_weights()

# Set to evaluation mode
model.eval()

# Create dummy inputs
input_image = torch.rand(1, 3, 512, 512)
input_sigma = torch.rand(1)

# Convert to ONNX
torch.onnx.export(
    model,
    (input_image, input_sigma),
    'FFDNet.onnx',
    verbose=False,
    input_names=['my_input', 'my_sigma'],
    output_names=['my_output'])
```

Some notes:

- The input PyTorch model is stored in the 'FFDNet.pth' file and the output ONNX model will be stored in the 'FFDNet.onnx' file.
- The `model.init_weights()` call is only necessary in my version of this particular model.
- Even if the PyTorch model accepts images of any size, ONNX needs to use a dummy input image with a fixed size to create the ONNX model. In this case, I've chosen to use a 512x512 image.
- The values composing the dummy imputs (image and sigma) are random as ONNX is using them only to go through the graph and deduce the graph structure.
- In the ONNX model, `my_input`, `my_sigma` and `my_output` respectively denotes the input RGB image, the sigma parameter (noise variance), and the output / denoised RGB image.


## ONNX to CoremML conversion: MultiArray version

### Conversion

The conversion from an ONNX model to a CoreML model is very simple:

```python
from onnx_coreml import convert
model = convert(model='FFDNet.onnx', minimum_ios_deployment_target='13')
model.save('FFDNet.mlmodel')
```

Using `coremltools`, one can display the specifications of the model.

```
input {
  name: "my_input"
  type {
    multiArrayType {
      shape: 1
      shape: 3
      shape: 512
      shape: 512
      dataType: FLOAT32
    }
  }
}
input {
  name: "my_sigma"
  type {
    multiArrayType {
      shape: 1
      dataType: FLOAT32
    }
  }
}
output {
  name: "my_output"
  type {
    multiArrayType {
      shape: 1
      shape: 3
      shape: 512
      shape: 512
      dataType: FLOAT32
    }
  }
}
metadata {
  userDefined {
    key: "coremltoolsVersion"
    value: "3.3"
  }
}
```

Some notes:

- Input and output names given during the PyTorch to ONNX conversion were transfered during the ONNX to CoreML conversion. Inputs are `my_input` and `my_sigma`, and output is `my_output`.
- `my_input` and `my_output` are of type `multiArrayType` which corresponds to the [MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray) object in CoreML.
- `my_input` and `my_output` are 4 dimensionnal array with fixed shape. The dimensions corresponds respectively to the mini-batch size, the number of channels, the image height and width.
- All values are coded on `float32`.


### Inference using coremltools

It's always wise to test the inference in Python before starting the integration process in iOS / macOS. Hopefully `coremltools` allows us to do this very simply:

```python
import coremltools
from PIL import Image
import numpy as np

# Load the CoreML model
model = coremltools.models.MLModel('FFDNet.mlmodel')
print(model.visualize_spec)

# Read input image
input_image = Image.open('input_512.png').convert('RGB')

# Convert the PIL image into a Numpy array
input_image = np.array(input_image)
input_image = input_image.transpose(2, 0, 1)       # Reorganise to channels x height x width
input_image = np.float32(input_image / 255.)       # Convert values in [0,1]
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# Sigma value
sigma = np.full(1, 0.1, np.float32)

# Prediction
prediction = model.predict({'my_input': input_image, 'my_sigma': sigma})

# Get the output image
output_image = prediction['my_output']

# Convert the Numpy array into a PIL image
output_image = np.squeeze(output_image, axis=0)  # Remove batch dimension
output_image = output_image * 255.               # Convert values in [0,255]
output_image = np.round(output_image).clip(0., 255.).astype(np.uint8)
output_image = output_image.transpose(1, 2, 0)   # Reorganise to height x width x channels
output_image = Image.fromarray(output_image)

# Save the output image
output_image.save('output_512.png')
```

The inference is pretty straight forward. `coremltools` accepts Numpy arrays since MLMutliArray are not available on Python. The most complicated part of this code is the image conversion (add / remove mini-batch dimension, conversion uint8 / float32).

I insist on the fact that the generated CoreML model only accepts images of size 512x512. It is possible to add flexibility to the model's input and ouput arrays using `coremltools` and more specifically `flexible_shape_utils`. But in my experience, I had troubles using this feature. It seemed to work with a regular convet but the custom layers used in FFDNet made the inference to crash. Note to self: fix this!


## ONNX to CoremML conversion: Image version

### Conversion

The conversion of an iOS / macOS image (e.g. [UIImage](https://developer.apple.com/documentation/uikit/uiimage)) into a MLMultiArray (and conversly) is not straightforward. It is possibile to modify the model during the conversion to allow to manipulate images directly, which will greatly simplify the integration process.

```python
from onnx_coreml import convert

model = convert(
    model='FFDNet.onnx',
    image_input_names=['my_input'],
    image_output_names=['my_output'],
    preprocessing_args={'image_scale': 1./255.},
    deprocessing_args={'image_scale': 255.},
    minimum_ios_deployment_target='13')

model.save('FFDNet_image.mlmodel')
```

Using coremltools, one can display the specifications of the model.

```
input {
  name: "my_input"
  type {
    imageType {
      width: 512
      height: 512
      colorSpace: RGB
    }
  }
}
input {
  name: "my_sigma"
  type {
    multiArrayType {
      shape: 1
      dataType: FLOAT32
    }
  }
}
output {
  name: "my_output"
  type {
    imageType {
      width: 512
      height: 512
      colorSpace: RGB
    }
  }
}
metadata {
  userDefined {
    key: "coremltoolsVersion"
    value: "3.3"
  }
}
```

Some notes:

- Using `image_input_names` and `image_output_names` in the conversion function, I've indicated to CoreML that `my_input` and `my_output` are actual images. One can see in the specifications that their type changed to `imageType`. This means I will no longer need to manipulate MLMultiArray in Objective-C or Swift. Instead, I will manipuate [CVPixelBuffer](https://developer.apple.com/documentation/corevideo/cvpixelbuffer-q2e). My life is easier already.
- The `preprocessing_args ` and `deprocessing_args` arguments allow to specify a preprocessing and postprocessing functions before and after the network. In my case, it allowed me to scale the image values from [0,255] to [0,1] before being sent to the network, and to scale the output values back from [0,1] to [0,255]. The conversion utin8 / float32 is also completely automatic. Note that more options are available for the `preprocessing_args ` and `deprocessing_args` arguments. See the [official documention](https://github.com/onnx/onnx-coreml).

### Inference using coremltools

The inference using coremltools becomes straightforward:

- The CoreML model manipulates directly PIL images.
- No uint8 / float32 and [0,255] / [0,1] conversions needed.

```python
import coremltools
from PIL import Image
import numpy as np

# Load the CoreML model
model = coremltools.models.MLModel('FFDNet_image.mlmodel')
print(model.visualize_spec)

# Read input image
input_image = Image.open('input_512.png').convert('RGB')

# Sigma value
sigma = np.full(1, 0.1, np.float32)

# Prediction
prediction = model.predict({'my_input': input_image, 'my_sigma': sigma})

# Get the output image
output_image = prediction['my_output']

# Save the output image
output_image.save('output_512_image.png')
```