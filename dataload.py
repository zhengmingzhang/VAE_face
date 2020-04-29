import os
import numpy as np
import torch
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors. numpy image: H x W x C, torch image: C X H X W
    """

    def __call__(self, image, invert_arrays=True):

        if invert_arrays:
            image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)
class CelebaDataset(Dataset):

    def __init__(self, root_dir, im_name_list, resize_dim, transform=None):
        self.root_dir = root_dir
        self.im_list = im_name_list
        self.resize_dim = resize_dim
        self.transform = transform

    def __len__(self):
        return len(self.im_list)

    # def __getitem__(self, idx):
    #     im = Image.open(os.path.join(self.root_dir, self.im_list[idx]))
    #     im = np.array(im)
    #     im = cv2.resize(im, self.resize_dim, interpolation=cv2.INTER_AREA)
    #     im = im / 255
    #     if self.transform:
    #         im = self.transform(im)

    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.root_dir, self.im_list[idx]))
        im = np.array(im)
        im = np.array(Image.fromarray(im).resize((self.resize_dim)))
        im = im / 255

        if self.transform:
            im = self.transform(im)

        return im


if __name__ == "__main__":
    root_dir = "./celeba_select"
    image_files = os.listdir(root_dir)
    train_dataset = CelebaDataset(root_dir, image_files[:5000], (64, 64), transforms.Compose([ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=144, num_workers=1, shuffle=True)
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data.type(torch.FloatTensor))
        print(data.shape)