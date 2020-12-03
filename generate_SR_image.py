import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from CONFIG import *
import os
from model import Generator


def gen_save_img(lr_path, name_of_img, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_name = lr_path + name_of_img
    save_name = save_path +'SR_'+ name_of_img
    image = Image.open(img_name)
    with torch.no_grad():
        image_tensor = Variable(ToTensor()(image)).unsqueeze(0)#.to(device)
        image_tensor = image_tensor.to(device)
        out = model_G(image_tensor)
        out_image = ToPILImage()(out[0].data.cuda())
        out_image.save(save_name)


if __name__ == '__main__':
    LR_PATH = TRAIN_HR_DIR+'/'
    PATH_G = PATH_MODEL_G
    SAVE_PATH = SR_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", str(device))
    model_G = Generator().to(device)
    checkpoint = torch.load(PATH_G)

    model_G.eval()

    name_of_image = '0001x4.png'
    gen_save_img(LR_PATH, name_of_image, SAVE_PATH)
