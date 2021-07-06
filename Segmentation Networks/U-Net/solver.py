import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from torchvision import transforms
from torchvision.transforms import ToTensor
import time
from tensorboardX import SummaryWriter
import cv2
import time
from copy import deepcopy
from torchvision.models import resnet50
torch.backends.cudnn.deterministic = True
class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, gpus):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
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
        self.result_path = config.result_path
        self.prediction_path = config.prediction_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.gpus = gpus
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
        # self.unet = torch.nn.DataParallel(self.unet)
        self.unet = torch.nn.DataParallel(self.unet.cuda(), device_ids=self.gpus, output_device=self.gpus[0])
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
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img
    def load_pretrain_model(self):
        state_path = '/home/srh/Image_Segmentation/pretrained_models/cem500k_mocov2_resnet50_200ep_pth.tar'
        state = torch.load(state_path)
        print(list(state.keys()))
        # ['epoch', 'arch', 'state_dict', 'optimizer', 'norms']

        state_dict = state['state_dict']
        resnet50_state_dict = deepcopy(state_dict)
        for k in list(resnet50_state_dict.keys()):
            # only keep query encoder parameters; discard the fc projection head
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                resnet50_state_dict[k[len("module.encoder_q."):]] = resnet50_state_dict[k]

            # delete renamed or unused k
            del resnet50_state_dict[k]

        # as before we need to update parameter names to match the UNet model
        # for segmentation_models_pytorch we simply and the prefix "encoder."
        # format the parameter names to match torchvision resnet50
        unet_state_dict = deepcopy(resnet50_state_dict)
        for k in list(unet_state_dict.keys()):
            unet_state_dict['encoder.' + k] = unet_state_dict[k]
            del unet_state_dict[k]

        self.unet.load_state_dict(unet_state_dict, strict=False)

    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#
        # load pre-train from other data
        self.load_pretrain_model()


        # load pre-train from former epoch
        # unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
        unet_path = ''
        # writer = SummaryWriter(log_dir='U-Net-log')

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
        else:
            # Train for Encoder
            lr = self.lr

            for epoch in range(self.num_epochs):
                start = time.time()

                self.unet.train(True)
                epoch_loss = 0

                acc = 0.	# Accuracy
                SE = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                length = 0

                for i, (images, GT, _) in enumerate(self.train_loader):
                    # GT : Ground Truth

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    # print("GT shape:   ",GT.shape)

                    # SR : Segmentation Result
                    SR = self.unet(images)
                    # print("out shape:     ",SR.shape)
                    SR_probs = F.sigmoid(SR)
                    # print("sigmoid shape:      ",SR_probs.shape)
                    SR_flat = SR_probs.view(SR_probs.size(0),-1)
                    # print("out view shape:      ",SR_flat.shape)

                    GT_flat = GT.view(GT.size(0),-1)
                    # print("GT view shape:      ",GT_flat.shape)
                    loss = self.criterion(SR_flat,GT_flat)
                    epoch_loss += loss.item()

                    # # write to web
                    # writer.add_scalar('loss', np.mean(np.nan_to_num(epoch_loss)), i)
                    # writer.add_image('image', images[0], epoch)
                    # writer.add_image('gt', GT[0], epoch)
                    # writer.add_image('prediction', SR_probs[0] * 255., epoch)


                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)
                    length += images.size(0)

                print('Training epoch {} times is {}: '.format(str(epoch),time.time()-start))

                # for each epoch, print a score
                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length
                # writer.add_scalars('Scores', {'acc': acc, 'SE': SE, "SP": SP, "PC":PC,"F1":F1,"JS":JS,"DC":DC},
                #                    epoch)

                # Print the log info
                print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                      epoch+1, self.num_epochs, \
                      epoch_loss,\
                      acc,SE,SP,PC,F1,JS,DC))

                # Decay learning rate
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decay learning rate to lr: {}.'.format(lr))


                #===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.	# Accuracy
                SE = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                length=0
                for i, (images, GT, _) in enumerate(self.valid_loader):

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = F.sigmoid(self.unet(images))
                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)

                    length += images.size(0)
                # validation scores
                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))

                '''
                torchvision.utils.save_image(images.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
                '''

                # # Save Best U-Net model
                # if unet_score > best_unet_score:
                #     best_unet_score = unet_score
                #     best_epoch = epoch
                #     best_unet = self.unet.state_dict()
                #     print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                #     torch.save(best_unet,unet_path)
                unet_path = os.path.join(self.model_path, '%s-%d.pkl' % (self.model_type, epoch))

                new_unet = self.unet.state_dict()
                torch.save(new_unet, unet_path)

                f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
                f.close()

    def test(self):
        unet_path = os.path.join(self.model_path, 'U_Net-150-0.0002-83-0.2629.pkl')
        save_path = self.prediction_path + self.model_type + '/' + 'U_Net-150-0.0002-83-0.2629/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.build_model()
        self.unet.load_state_dict(torch.load(unet_path))

        self.unet.train(False)
        self.unet.eval()

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        F1 = 0.  # F1 Score
        JS = 0.  # Jaccard Similarity
        DC = 0.  # Dice Coefficient
        length = 0
        for i, (images, GT, filename) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images))

            acc += get_accuracy(SR, GT)
            SE += get_sensitivity(SR, GT)
            SP += get_specificity(SR, GT)
            PC += get_precision(SR, GT)
            F1 += get_F1(SR, GT)
            JS += get_JS(SR, GT)
            DC += get_DC(SR, GT)

            #========save prediction=========#
            out = SR.cpu().detach().numpy()
            # print(GT[0,0,:,:])
            # print(out[0,0,:,:])
            # exit()
            for k in range(len(filename)):
                if not os.path.exists(save_path+filename[k].split('/')[0]):
                    os.makedirs(save_path+filename[k].split('/')[0])
                cv2.imwrite(save_path+filename[k],out[k,0,:,:]*255)

            length += images.size(0)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        unet_score = JS + DC

        f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow([self.model_type, 'acc', 'SE', 'SP', 'PC', 'F1', 'JS', 'DC'])
        wr.writerow([self.model_type, acc, SE, SP, PC, F1, JS, DC])

        f.close()


