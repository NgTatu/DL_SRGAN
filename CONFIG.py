HR_CROP_SIZE = 100
LR_CROP_SIZE = HR_CROP_SIZE/4
UPSCALE_FACTOR = 4
NUM_EPOCHS = 20
BATCH_SIZE = 8

TRAIN_HR_DIR = '/home/tuan/Documents/VBDI/SRGAN/data/DIV2K_train_HR'
TRAIN_LR_DIR = '/home/tuan/Documents/VBDI/SRGAN/data/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4'

VAL_HR_DIR = '/home/tuan/Documents/VBDI/SRGAN/data/DIV2K_valid_HR'
VAL_LR_DIR = '/home/tuan/Documents/VBDI/SRGAN/data/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4'

### USE THIS FOR COLAB ####
# TRAIN_HR_DIR = '/content/drive/My Drive/srgan/srgan_tensorlayer/data/DIV2K_train_HR'
# TRAIN_LR_DIR = '/content/drive/My Drive/srgan/srgan_tensorlayer/data/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4'
#
# VAL_HR_DIR = '/content/drive/My Drive/srgan/srgan_tensorlayer/data/DIV2K_valid_HR'
# VAL_LR_DIR = '/content/drive/My Drive/srgan/srgan_tensorlayer/data/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4'