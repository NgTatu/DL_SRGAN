from os import listdir
from os.path import join
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size%upscale_factor)


def data_hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])


def data_lr_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size // upscale_factor),
        ToTensor()
    ])


class DataSetFromFolder(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor):
        super(DataSetFromFolder, self).__init__()
        self.hr_image_filenames = [join(hr_data_dir, x) for x in np.sort(listdir(hr_data_dir)) if is_image_file(x)]
        self.lr_image_filenames = [join(lr_data_dir, x) for x in np.sort(listdir(lr_data_dir)) if is_image_file(x)]
        crop_size = calculate_crop_size(crop_size, upscale_factor) #upscale_factor is 4
        self.hr_transform = data_hr_transform(crop_size)
        self.lr_transform = data_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_image_filenames[index]))
        lr_image = self.lr_transform(Image.open(self.lr_image_filenames[index]))
        return hr_image, lr_image

    def __len__(self):
        return len(self.hr_image_filenames)
