import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import os
import sys
from torchvision import transforms
import torchvision

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_pickle(f):
    if sys.version_info[0] == 2:
        return pickle.load(f)
    elif sys.version_info[0] == 3:
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(sys.version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y


###  suggested reference: https://pytorch.org/tutorials/
# recipes/recipes/custom_dataset_transforms_loader.html?highlight=dataloader
# functions to show an image

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True, N=None):
        """
                Initializes a CIFAR10_loader instance.

                Args:
                    root (str): Root directory of the CIFAR-10 dataset.
                    train (bool, optional): If True, loads the training data. If False, loads the test data. Defaults to True.
                    transform (callable, optional): A transform to apply to the data. Defaults to None.
                    N (int, optional): Maximum number of samples per class. Defaults to None.
        """
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.N = N
        self.data_update()

    def data_update(self):
        assert len(self.data) == len(self.targets)
        label_mapping = {0: 0, 2: 1, 8: 2, 7: 3, 1: 4}

        new_data = []
        new_targets = []
        class_counter = np.zeros(5)

        for item in range(len(self.data)):
            label = self.targets[item]
            if label in label_mapping:
                new_label_value = label_mapping[label]
                if self.N is None or class_counter[new_label_value] < self.N:
                    # Increment the class_counter and add the data and new label
                    class_counter[new_label_value] += 1
                    new_data.append(self.data[item])
                    new_targets.append(new_label_value)
            if self.N is not None and np.all(class_counter == self.N):
                break

        self.data = np.asarray(new_data)
        self.targets = np.asarray(new_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = CIFAR10('./data', train=True, transform=transform, download=True, N=None)
    print(f"Train data: {train_set.data.shape}")
    print(f"Train labels: {train_set.targets.shape}")

    trainloader = DataLoader(train_set, batch_size=4,
                             shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(images.shape)

    test_set = CIFAR10('./data', train=False, transform=transform, download=True, N=None)
    print(f"Test data: {test_set.data.shape}")
    print(f"Test labels: {test_set.targets.shape}")
