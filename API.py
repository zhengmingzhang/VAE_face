import torch
import cv2
import random
from model import AutoEncoder
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from dataload import *

class face_AE():

    def __init__(self):
        self.auto: AutoEncoder = torch.load("./autoencoder.pkl",map_location='cpu')
        self.auto.eval()

    def get_img(self, img_path):
        im = Image.open(img_path)
        im = np.array(im)
        im = np.array(Image.fromarray(im).resize((64,64)))
        im = im / 255
        im = im.transpose((2, 0, 1))
        im = torch.from_numpy(im)
        im = im.unsqueeze(0)
        im = im.float()
        return im

    def get_feature(self, img):
        mu, log_var = self.auto.encode(img)
        feature = self.auto.reparameterize(mu, log_var)
        return feature

    def invtrans(self, feature):
        img_re = self.auto.decode(feature).cpu()
        rand_name = random.randint(0,100)
        save_image(img_re, './imgs/' + str(rand_name) + '.png')
        return img_re

    def get_rand_face(self, length=128):
        sample = Variable(torch.randn(64, length))
        sample = self.auto.decode(sample).cpu()
        rand_name = random.randint(0, 100)
        save_image(sample.data.view(64, 3, 64, 64), './imgs/' + str(rand_name) + '.png')

    def get_reconstruction(self):
        n = 10
        root_dir = "./celeba_select"
        image_files = os.listdir(root_dir)
        test_dataset = CelebaDataset(root_dir, image_files[:10], (64, 64), transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=10, num_workers=1, shuffle=True)
        for i, data in enumerate(test_loader):
            data = Variable(data.type(torch.FloatTensor), volatile=True)
            print(data.shape)
            feature = self.get_feature(data)
            recon_batch = self.auto.decode(feature).cpu()
            print(recon_batch.shape)
            if i == 0:
                comparison = torch.cat([data[:n],
                                        recon_batch.view(10, 3, 64, 64)[:n]])
                save_image(comparison.data.cpu(),
                           './imgs/reconstruction' +  '.png', nrow=n)

if __name__ == "__main__":
    face = face_AE()
    img_path = "./celeba_select/000181.jpg"
    img = face.get_img(img_path)
    face.get_rand_face()
    face.get_reconstruction()


