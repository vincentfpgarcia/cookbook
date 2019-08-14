# Pillow

## Forewords

Pillow is the friendly PIL fork by Alex Clark and Contributors. PIL is the Python Imaging Library by Fredrik Lundh and Contributors. See [here](https://pillow.readthedocs.io/en/stable/) for a complete information about Pillow.

In this page, I present the most basic functionalities provided by the `Image` module. Pillow contains many modules that will let you manipulate images. Please read the official documentation.


## Open an image

Simple image read and properties cheking.

```python
from PIL import Image

# Read the input image
img = Image.open("guinness.jpg")

# Display basic properties
print("What   : %s" % type(img))
print("Width  : %d" % img.width)
print("Height : %d" % img.height)
print("Mode   : %s" % img.mode)
```

The output is:

```
What   : <class 'PIL.JpegImagePlugin.JpegImageFile'>
Width  : 960
Height : 540
Mode   : RGB
```


## Display and image

You can display an image directly using Pillow:

```python
from PIL import Image

# Read the input image
img = Image.open("guinness.jpg")

# Display the image
img.show()
```


## Access image pixels

When it comes to image processing, knowing how to access and modify pixels is unavoidable.

```python
from PIL import Image

# Read the input image
img = Image.open("guinness.jpg")

# Access the pixel value at position x=50, y=100
print(img.getpixel((100,50)))

# Modify the pixel value at position x=50, y=100
img.putpixel((100,50), (50, 100, 200))

# Verify the pixel value at position x=50, y=100
print(img.getpixel((100,50)))
```

The output is:

```
(146, 138, 136)
(50, 100, 200)
```


## Save and image

Another essential function to know:

```python
from PIL import Image

# Read the input image
img = Image.open("guinness.jpg")

# Save the image
img.save("test.png")
```


## Create a new image

Here, we are creating an image of size 256x256. To verify that we can access the image, we fill it with a red horizontal gradient and blue vertical gradient. The green channel is set to 0.

```python
from PIL import Image

# Create a new image of size 256 x 256
img = Image.new("RGB", (256, 256))

# Fill the image with:
# - Red   = horizontal gradient
# - Green = 0
# - Blue  = vertical gradient
for y in range(256):
    for x in range(256):
        img.putpixel((x,y), (x, 0, y))
        
# Show the image
img.show()
```

The displayed image should look like this:

![Result](pil_test4.png)

## Interaction with Numpy

Pillow interacts well with Numpy. Numpy is indeed very convenient for some operations. Bellow, we open an image with Pillow, convert it into an Numpy array, modify it using Numpy, and converti it back into a Pillow image.

```python
import numpy as np
from PIL import Image

# Read the input image
img_pil = Image.open("guinness.jpg")

# Convert the PIL image into a Numpy array
img_np = np.array(img_pil)

# Display basic properties
print("Width  : %d" % img_np.shape[1])
print("Height : %d" % img_np.shape[0])
print("Depth  : %d" % img_np.shape[2])
print("Type   : %s" % img_np.dtype.name)

# Set the red channel to 255 for all pixels
img_np[:,:,0] = 255

# Convert the Numpy image into a PIL image
img_pil2 = Image.fromarray(img_np)

# Display the image
img_pil2.show()
```

The output is:

```
Width  : 960
Height : 540
Depth  : 3
Type   : uint8
```