import torch
from torch import nn
from torchvision.models.vgg import vgg19
adv_coef = 1e-1
per_coef = 1e-1

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        self.vgg_loss = nn.Sequential(*list(vgg.features)).eval()
        for param in self.vgg_loss.parameters():
            param.requires_grad = False
        self.loss_network = self.vgg_loss
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        '''

        :param out_labels: label of output image (real or fake) = D(G(z))
        :param out_images: = G(z)
        :param target_images: real HR images
        :return:
        '''
        # Adversarial loss
        # adversarial_loss = torch.mean(1 - out_labels)
        adversarial_loss = torch.mean(1-torch.log(out_labels))
        # Perception loss
        perception_loss = self.mse_loss(self.vgg_loss(out_images), self.vgg_loss(target_images))
        # Image loss
        image_loss = self.mse_loss(out_images, target_images)
        print('adversarial_loss: %.4f - perception_loss: %.4f' % (adv_coef*adversarial_loss.item(), per_coef*perception_loss.item()))
        # print('adversarial_loss: ' + str(0.001*adversarial_loss.item()) + '   -   '+'perception_loss: '+str(0.006*perception_loss.item()))
        return image_loss + adv_coef*adversarial_loss + per_coef*perception_loss
