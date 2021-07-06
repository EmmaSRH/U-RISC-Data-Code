import argparse
import os
from tqdm import tqdm
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import torch
import cv2
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def compute_integrated_gradient(batch_x, model,px,py):
    mean_grad = 0
    n = 100
    batch_blank=torch.zeros_like(batch_x)
    for i in tqdm(range(1, n + 1)):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)[0,0,px,py]
        (grad,) = torch.autograd.grad(y, x)
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad
    return integrated_gradients

def plot_images(images,output,gdmap,gt,output_path):
    fig, axs = plt.subplots(1,4)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    axs[0].imshow(images)
    axs[2].imshow(output,cmap = plt.cm.gray)
    axs[1].imshow(gdmap)
    axs[3].imshow(gt)
    fig.tight_layout()
    plt.savefig(output_path)

def plot_grad(data):
    abs_max=max(data.max(),-data.min())
    img=np.ones((512,512,3))*255.0
    img[:,:,0]+=data*(data<0)/abs_max*255
    img[:,:,1]+=data*(data<0)/abs_max*255
    img[:,:,1]-=data*(data>0)/abs_max*255
    img[:,:,2]-=data*(data>0)/abs_max*255
    return img.astype(np.uint8)

def add_point(arr,j,k):
    for u in range(-2, 2):
        for v in range(-2, 2):
            if j + u >= 0 and j + u < 512 and k + v >= 0 and k + v < 512:
                arr[j + u, k + v, 0] = 0
                arr[j + u, k + v, 1] = 255
                arr[j + u, k + v, 2] = 0
    return arr

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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_name', type=str, default='512-149.pkl', help='U_Net-149.pkl')
    parser.add_argument('--prediction_path', type=str, default='/raid/srh/IS/urisc/prediction/')
    # parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--test_path', type=str, default='/raid/srh/Image_Segmentation/urisc_data/')
    parser.add_argument('--result_path', type=str, default='/raid/srh/Image_Segmentation/result/')

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
        # self.model_path = config.model_path
        self.model_name = config.model_name
        self.result_path = config.result_path
        self.prediction_path = config.prediction_path
        self.test_path = config.test_path


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
        # unet_path = os.path.join(self.model_path, self.model_name)
        unet_path = self.model_name
        save_path = self.prediction_path + self.model_type + '/' + self.model_name.split('.')[0] +'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.build_model()
        print("build_success")
        self.unet.load_state_dict(torch.load(unet_path))

        self.unet.train(False)
        self.unet.eval()

        for i, (images, GT, filename) in enumerate(self.test_loader):
            if filename[0]!='16.png':
                continue
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images))
            out = SR.cpu().detach().numpy()
            # print('out.shape:       ', out.shape)
            print(filename[0])


            # draw by pixel
            for j in range(4,512,10):
                for k in range(4,512,10):
                    # ig
                    ig=compute_integrated_gradient(images,self.unet,j,k)
                    ig=plot_grad(ig[0,0].cpu().detach().numpy())
                    # ig = add_point(ig, j, k)
                    save_ig_path = '/raid/srh/IS/ig/urisc/' + filename[0].split('.')[0] + '/'
                    if not os.path.exists(save_ig_path):
                        os.mkdir(save_ig_path)
                    cv2.imwrite(save_ig_path+str(j)+'_'+str(k)+'.png', ig)
                    # print('ig.shape:      ',ig.shape)

                    # pic
                    pic=np.zeros((512,512,3)).astype(np.uint8)
                    pic[:,:,0]=out[0,0,:,:]*255
                    pic[:,:,1]=out[0,0,:,:]*255
                    pic[:,:,2]=out[0,0,:,:]*255

                    save_pre_path = '/raid/srh/IS/prediction/urisc/'
                    if not os.path.exists(save_pre_path):
                        os.mkdir(save_pre_path)
                    cv2.imwrite(save_pre_path + filename[0], pic)

                    pic = add_point(pic, j, k)

                    # print('pic.shape:      ',pic.shape)

                    # input
                    input_img = cv2.imread(self.test_path+filename[0])
                    input_img = cv2.resize(input_img, (512,512))
                    input_img = add_point(input_img, j, k)

                    # gt
                    gt = cv2.imread(self.test_path+filename[0].replace('.png','.jpg'))
                    gt = cv2.resize(gt, (512,512))
                    gt = add_point(gt, j, k)

                    save_pic_path = '/raid/srh/IS/urisc/'+filename[0].split('.')[0]+'/'
                    if not os.path.exists(save_pic_path):
                        os.mkdir(save_pic_path)
                    plot_images(input_img,pic,ig,gt,save_pic_path+str(j)+'_'+str(k)+'.png')

            # break




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
    print("start_test")
    solver.test()

