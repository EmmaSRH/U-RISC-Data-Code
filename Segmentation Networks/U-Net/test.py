import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import torch
import cv2
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from torch import optim
import torch.nn.functional as F

def options():
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_name', type=str, default='U_Net-91.pkl', help='U_Net-149.pkl')
    parser.add_argument('--prediction_path', type=str, default='./pre-prediction-5-21/')
    parser.add_argument('--model_path', type=str, default='./use_pre_models')
    parser.add_argument('--test_path', type=str, default='/mnt/srh/U-RISC-DATASET/patchs/1024/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    config = parser.parse_args()

    return config

class Solver(object):
    def __init__(self, config, test_loader):

        # Data loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.model_name = config.model_name
        self.result_path = config.result_path
        self.prediction_path = config.prediction_path


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=3,output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)


        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)
        self.unet = torch.nn.DataParallel(self.unet)
        self.unet = self.unet.cuda()

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cuda()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()


    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img

    def test(self):
        unet_path = os.path.join(self.model_path, self.model_name)
        save_path = self.prediction_path + self.model_type + '/' + self.model_name.split('.')[0] +'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.build_model()
        self.unet.load_state_dict(torch.load(unet_path))

        self.unet.train(False)
        self.unet.eval()

        for i, (images, GT, filename) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images))

            #========save prediction=========#
            out = SR.cpu().detach().numpy()

            for k in range(len(filename)):
                if not os.path.exists(save_path+filename[k].split('/')[0]):
                    os.makedirs(save_path+filename[k].split('/')[0])
                cv2.imwrite(save_path+filename[k],out[k,0,:,:]*255)




if __name__ == '__main__':
    config = options()
    cudnn.benchmark = True

    lr = random.random() * 0.0005 + 0.0000005
    augmentation_prob = random.random() * 0.7
    epoch = random.choice([100, 150, 200, 250])
    decay_ratio = random.random() * 0.8
    decay_epoch = int(epoch * decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    #==========  load test data  =====#
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.)


    solver = Solver(config, test_loader)

    solver.test()

