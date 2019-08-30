def example0():
    
    import torch
    # import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='~/datasets',
                                            train=True,
                                            download=True,
                                            transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='~/datasets',
                                       train=False,
                                       download=True,
                                       transform=transform)

    testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)

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
    from PIL import Image

    # Access the training set
    datasets_dir = '~/datasets'
    trainset = torchvision.datasets.CIFAR10(root=datasets_dir, train=True, download=True)

    # Pick an image in the dataset
    idx = 20

    # Access the corresponding image label
    label = trainset.targets[idx]
    label_str = trainset.classes[label]
    print("Index = %d" % idx)
    print("Label = %d -> %s" % (label, label_str))

    # Access the image and display it
    img_np = trainset.data[idx,:,:,:]
    print(img_np.dtype)
    img_pil = Image.fromarray(img_np)
    img_pil.show()


def example3():
    
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from PIL import Image
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
                                              shuffle=True,
                                              num_workers=2)

    # Access the first image batch
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # imshow(torchvision.utils.make_grid(images))

    # print('make_grid', type(torchvision.utils.make_grid(images)))

    # print(type(images))
    # print(images.size())
    # return

    # Convert the torch.Tensor into a Numpy array
    images_np = images.numpy()
    print("Array size: %s" % str(images_np.shape))
    print("Array type: %s" % images_np.dtype)

    # tmp = np.transpose(images_np[0,:,:,:], (1, 2, 0))
    # tmp = (tmp*255).astype(np.uint8)
    # plt.imshow(tmp)
    # plt.show()

    tmp = np.zeros(256,256)
    print(tmp.shape)


    # f, ax = plt.subplots(1, 4)

    # for i in range(4):
    #     img = images_np[i,:,:,:]
    #     img = np.transpose(img, (1, 2, 0))

    #     ax[0].imshow(img)
    # plt.show()

    # tmp = np.transpose(images_np[0,:,:,:], (1, 2, 0))
    # tmp = (tmp*255).astype(np.uint8)
    # img_pil = Image.fromarray(tmp)
    # img_pil.show()


    # print(dir(images))
    # print(type(images_np))
    # print(images_np.shape)

    # # Pick an image in the dataset
    # idx = 20

    # # Access the corresponding image label
    # label = trainset.targets[idx]
    # label_str = trainset.classes[label]
    # print("Index = %d" % idx)
    # print("Label = %d -> %s" % (label, label_str))

    # Access the image and display it
    # img_np = trainset.data[idx,:,:,:]
    # print(type(img_np))
    # img_pil = Image.fromarray(img_np)
    # img_pil.show()




if __name__ == '__main__':
    # example0()
    # example1()
    # example2()
    # example3()

    # import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # tmp = np.zeros((256,256,3), dtype=np.uint8)

    matplotlib.use('MacOSX')

    # print(dir(matplotlib))

    # print(tmp.shape)
    # print(tmp.dtype)
    plt.figure()
    # plt.imshow(tmp)
    plt.show()
