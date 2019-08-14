def test0():

    from PIL import Image

    # Read the input image
    img = Image.open("guinness.jpg")

    # Display basic properties
    print("What   : %s" % type(img))
    print("Width  : %d" % img.width)
    print("Height : %d" % img.height)
    print("Mode   : %s" % img.mode)


def test1():

    from PIL import Image

    # Read the input image
    img = Image.open("guinness.jpg")

    # Display the image
    img.show()


def test2():

    from PIL import Image

    # Read the input image
    img = Image.open("guinness.jpg")

    # Access the pixel value at position x=50, y=100
    print(img.getpixel((100,50)))

    # Modify the pixel value at position x=50, y=100
    img.putpixel((100,50), (50, 100, 200))

    # Verify the pixel value at position x=50, y=100
    print(img.getpixel((100,50)))


def test3():
    
    from PIL import Image

    # Read the input image
    img = Image.open("guinness.jpg")

    # Save the image
    img.save("test.png")


def test4():

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


def test5():

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


if __name__ == '__main__':
    # test0()
    # test1()
    # test2()
    # test3()
    test4()
    # test5()

