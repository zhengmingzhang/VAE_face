import torch
import cv2
import random
from model import AutoEncoder


class face_AE():
    def __init__(self):
        self.auto: AutoEncoder = torch.load("./models/autoencoder.pkl",map_location='cpu')

    def get_img(self, img_path):
        image = []
        img = cv2.imread(img_path)
        if img is None:
            print('错误! 无法在该地址找到图片!')
        dim = (64, 64)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        image.append(img)
        img = torch.Tensor(image).permute(0, 3, 2, 1) / 255
        return img

    def get_feature(self, img):
        mu, log_var = self.auto.encode(img)
        feature = self.auto.reparameterize(mu, log_var)
        return feature

    def invtrans(self, feature):
        img_re = self.auto.decode(feature)
        print(img_re.shape)
        img_re = img_re.contiguous().permute(0, 3, 2, 1).detach().cpu().numpy()*255
        print(img_re[0].shape)
        rand_name = random.randint(0,100)
        cv2.imwrite("imgs/" + str(rand_name)+ ".jpg", img_re[0])
        return img_re

    def get_rand_face(self, length=128):
        random_feature = []
        for i in range(length):
            feature = random.uniform(-0.5, 0.5)
            feature = round(feature, 4)
            random_feature.append(feature)
        random_feature = torch.FloatTensor(random_feature)
        self.invtrans(random_feature)
        return random_feature
if __name__ == "__main__":
    face = face_AE()
    img_path = "./celeba_select/000281.jpg"
    img = face.get_img(img_path)
    feature = face.get_feature(img)
    print(feature.shape)
    #face.invtrans(feature)
    for i in range(30):
        face.get_rand_face()


