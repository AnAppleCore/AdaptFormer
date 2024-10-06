import os
import os.path

from shutil import move, rmtree

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url


class CUB200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
        self.filename = 'CUB_200_2011.tgz'

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'CUB_200_2011')):
            # import zipfile
            # zip_ref = zipfile.ZipFile(fpath, 'r')
            # zip_ref.extractall(root)
            # zip_ref.close()

            import tarfile
            tar_ref = tarfile.open(fpath, 'r')
            tar_ref.extractall(root)
            tar_ref.close()

            self.split()
        
        if self.train:
            fpath = os.path.join(root, 'CUB_200_2011', 'train')

        else:
            fpath = os.path.join(root, 'CUB_200_2011', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def split(self):
        train_folder = os.path.join(self.root, 'CUB_200_2011', 'train')
        test_folder = os.path.join(self.root, 'CUB_200_2011', 'test')

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        images = os.path.join(self.root, 'CUB_200_2011/images.txt')
        train_test_split = os.path.join(self.root, 'CUB_200_2011/train_test_split.txt')

        with open(images, 'r') as image:
            image_paths = image.readlines()
            with open(train_test_split, 'r') as f:
                i = 0
                for line in f:
                    image_path = image_paths[i]
                    image_path = image_path.replace('\n', '').split(' ')[-1]
                    class_name = image_path.split('/')[0]
                    src = os.path.join(self.root, 'CUB_200_2011/images', image_path)

                    if line.split(' ')[-1].replace('\n', '') == '1':
                        if not os.path.exists(train_folder + '/' + class_name):
                            os.mkdir(train_folder + '/' + class_name)
                        dst = train_folder + '/' + image_path
                    else:
                        if not os.path.exists(test_folder + '/' + class_name):
                            os.mkdir(test_folder + '/' + class_name)
                        dst = test_folder + '/' + image_path
                    
                    move(src, dst)
                    i += 1