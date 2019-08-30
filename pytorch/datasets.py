def example1():
    
    import torch
    import torchvision

    # Directory where datasets will be stored
    datasets_dir = '~/datasets'

    # Training set
    trainset = torchvision.datasets.CIFAR10(root=datasets_dir,
                                            train=True,
                                            download=True)

    # Test set
    testset = torchvision.datasets.CIFAR10(root=datasets_dir,
                                           train=False,
                                           download=True)

    # Access datasets properties
    train_shape = trainset.data.shape
    test_shape = testset.data.shape
    train_nb = train_shape[0]
    test_nb = test_shape[0]
    height = train_shape[1]
    width = train_shape[2]
    classes = trainset.classes
    
    print("Training set size : %d" % train_nb)
    print("Test set size     : %d" % test_nb)
    print("Image size        : %d x %d" % (height, width))
    print("List of classes")
    for val in classes:
      print("- %s" % val)


def example2():
    
    import torch
    import torchvision
    import matplotlib.pyplot as plt

    # Access the training set
    datasets_dir = '~/datasets'
    trainset = torchvision.datasets.CIFAR10(root=datasets_dir, train=True, download=True)

    # Pick an image in the dataset
    idx = 20

    # Access the corresponding image label
    label = trainset.targets[idx]
    label_str = trainset.classes[label]
    print("Image index : %d" % idx)
    print("Image label : %d (%s)" % (label, label_str))

    # Access the image and display it
    img = trainset.data[idx,:,:,:]
    print("Image type  : %s" % img.dtype)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def example3():
    
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    # Access the training set
    datasets_dir = '~/datasets'
    trainset = torchvision.datasets.CIFAR10(root=datasets_dir,
                                            train=True,
                                            download=True,
                                            transform=transforms.ToTensor())

    # Define the data loader
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=False,
                                              num_workers=2)

    # Access the first image batch
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Convert the torch.Tensor into a Numpy array
    images_np = images.numpy()
    labels_np = labels.numpy()

    # Print batch shape
    print("Array shape : %s" % str(images_np.shape))
    print("Array type  : %s" % images_np.dtype)

    # Access the first image of the batch
    idx = 0
    img = np.transpose(images_np[idx,:,:,:], (1, 2, 0))
    lbl = labels_np[idx]
    print("Image index : %d" % idx)
    print("Image label : %d (%s)" % (lbl, trainset.classes[lbl]))

    # Display
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # example1()
    # example2()
    example3()
