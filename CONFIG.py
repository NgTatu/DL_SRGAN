HR_CROP_SIZE = 300
LR_CROP_SIZE = HR_CROP_SIZE/4
UPSCALE_FACTOR = 4
NUM_EPOCHS = 20
BATCH_SIZE = 8

MODEL_CHECKPOINT_PATH = '/content/drive/My Drive/srgan/model/'
# MODEL_CHECKPOINT_PATH = 'model/'
PATH_MODEL_G = MODEL_CHECKPOINT_PATH + 'G.pt'
PATH_MODEL_D = MODEL_CHECKPOINT_PATH + 'D.pt'
SR_PATH = '/content/drive/My Drive/srgan/SR_images/'

# TRAIN_HR_DIR = '/home/tuan/Documents/VBDI/SRGAN/data/DIV2K_train_HR'
# TRAIN_LR_DIR = '/home/tuan/Documents/VBDI/SRGAN/data/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4'
#
# VAL_HR_DIR = '/home/tuan/Documents/VBDI/SRGAN/data/DIV2K_valid_HR'
# VAL_LR_DIR = '/home/tuan/Documents/VBDI/SRGAN/data/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4'

### USE THIS FOR COLAB ####
TRAIN_HR_DIR = '/content/drive/My Drive/srgan/srgan_tensorlayer/data/DIV2K_train_HR'
TRAIN_LR_DIR = '/content/drive/My Drive/srgan/srgan_tensorlayer/data/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4'

VAL_HR_DIR = '/content/drive/My Drive/srgan/srgan_tensorlayer/data/DIV2K_valid_HR'
VAL_LR_DIR = '/content/drive/My Drive/srgan/srgan_tensorlayer/data/DIV2K_valid_LR_bicubic_X4/DIV2K_valid_LR_bicubic/X4'