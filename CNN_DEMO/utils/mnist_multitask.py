import torch
from torchvision import datasets, transforms
from PIL import Image


def load_multitask_dataset(namedataset='mnist_rotate', n_samples=1000, rotation=90, batch_size=100):

    if namedataset == 'mnist_rotate':

        DIR_DATASET = '~/data/mnist'

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(x.size(1)*x.size(2))),
        ])

        trainset = MNIST_ROTATE(DIR_DATASET, train=True, rotation=rotation, n_samples=n_samples, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = MNIST_ROTATE(DIR_DATASET, train=False, rotation=rotation, n_samples=n_samples, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

        classes = tuple(range(10))
        n_inputs = 784

    elif namedataset == 'mnist_permute':

        DIR_DATASET = '~/data/mnist'

        transform = transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(x.size(1)*x.size(2))),
        ])

        trainset = MNIST_PERMUTE(DIR_DATASET, train=True, n_samples=n_samples, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        # Make sure to use same permutation for traning and test set...
        testset = MNIST_PERMUTE(DIR_DATASET, permutation=trainset.permutation, train=False, n_samples=n_samples, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

        classes = tuple(range(10))
        n_inputs = 784

    else:
        raise ValueError('Dataset {} not recognized'.format(namedataset))

    return train_loader, test_loader, n_inputs


class MNIST_ROTATE(datasets.MNIST):

    def __init__(self, root, train=True, rotation=90, n_samples=1000, transform=None, target_transform=None, download=False):
        super(MNIST_ROTATE, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)
        self.rotation = rotation

        '''New torchvision convention:'''
        if not hasattr(self, 'data'):
            if self.train:
                self.data = self.train_data
                self.targets = self.train_labels
            else:
                self.data = self.test_data
                self.targets = self.test_labels

        self.data = self.data[:n_samples]
        self.targets = self.targets[:n_samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.rotate(self.rotation)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class MNIST_PERMUTE(datasets.MNIST):

    def __init__(self, root, permutation=None, train=True, n_samples=1000, transform=None, target_transform=None, download=False):
        super(MNIST_PERMUTE, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)
        '''New torchvision convention:'''
        if not hasattr(self, 'data'):
            if self.train:
                self.data = self.train_data
                self.targets = self.train_labels
            else:
                self.data = self.test_data
                self.targets = self.test_labels

        if permutation is None:
            self.permutation = torch.randperm(self.data.numel()//len(self.data)).long().view(-1)
        else:
            self.permutation = permutation

        size = self.data.size()
        self.data = self.data.view(size[0], -1).index_select(1, self.permutation).view(size)

        self.data = self.data[:n_samples]
        self.targets = self.targets[:n_samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
