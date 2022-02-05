#coding: utf-8
import os
from PIL import Image
from matplotlib import patches
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from nets.u_net import UNet
from nets.Four_layer_net import Four_layer
from main import train_one_epoch
from nets.msf_updec_sum_net import MSF_UpDec_ResUNet
from DataSet import load_train_val_data
from nets.ResUNetPlusPlus import ResUNetPlusPlus
from nets.SN_msf_updec_sum_net import SN_MSF_UpDec_ResUNet
from nets.MultiResUNet import MultiResUnet
import nets.segmentation_models_pytorch as smp
from nets.attention_unet import AttU_Net
import os
import time
from Losses import *

import warnings
warnings.filterwarnings("ignore")


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    dice_show = "%.3f" % (dice_pred)
    fig, ax = plt.subplots()
    plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    plt.imshow(predict_save * 255,cmap='gray')
    plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    plt.axis("off")
    # 去除图像周围的白边
    height, width = predict_save.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
    plt.savefig(save_path, dpi=300)
    plt.close()
    return dice_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()
    if model_type[0:4] == 'LBTW':
        output,a,b,c = model(input_img.cuda())
    else:
        output = model(input_img.cuda())
    pred_class = torch.where(output>0.35,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (640, 640))
    dice_pred_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    return dice_pred_tmp




if __name__ == '__main__':
    ## PARAMS
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # model_type = 'ResUNetPP'
    # model_type = 'MultiResUNet'
    # model_type = 'Attention_UNet'
    model_type = 'UNetPP'
    # model_type = 'UNet'
    # model_type = 'LBTWDS_MSF_UpDec_ResUNet_SN'
    data_path = '../Segmentation_dataset/' #path where dataset is stored


    ###############################################
    #      MA
    ###############################################
    # save_path = '/home/user/whn/IDRiD/Models/LBTWDS_MSF_UpDec_ResUNet_SN/save/models/Test_session_04.29 16h47/best_model.LBTW_WeightedDiceBCE--LBTWDS_MSF_UpDec_ResUNet_SN.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/ResUNetPP/save/models/Test_session_03.22 18h06/best_model.WeightedDiceBCE--ResUNetPP.pth.tar' #path where weights of our model was downloaded
    # save_path = '/home/user/whn/IDRiD/Models/MultiResUNet/save/models/Test_session_03.23 08h26/best_model.WeightedDiceBCE--MultiResUNet.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/UNetPP/save/models/Test_session_03.22 22h22/best_model.WeightedDiceBCE--UNetPP.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/UNet/save/models/Test_session_03.25 10h58/model.WeightedDiceBCE--UNet--99.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/Attention_UNet/save/models/Test_session_04.30 00h08/best_model.WeightedDiceBCE--Attention_UNet.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/MultiResUNet/save/models/Test_session_04.29 10h53/model.WeightedDiceBCE--MultiResUNet--109.pth.tar'
    ###############################################
    #      EX
    ###############################################
    # save_path = '/home/user/whn/IDRiD/Models/EX/ResUNetPP/save/models/Test_session_04.14 08h23/best_model.WeightedDiceBCE--ResUNetPP.pth.tar'
    save_path = '/home/user/whn/IDRiD/Models/EX/UNetPP/save/models/Test_session_04.14 21h36/best_model.WeightedDiceBCE--UNetPP.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/EX/LBTWDS_MSF_UpDec_ResUNet_SN/save/models/Test_session_04.27 08h29/best_model.LBTW_WeightedDiceBCE--LBTWDS_MSF_UpDec_ResUNet_SN.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/EX/UNet/save/models/Test_session_04.28 18h43/best_model.WeightedDiceBCE--UNet.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/EX/MultiResUNet/save/models/Test_session_04.28 22h58/best_model.WeightedDiceBCE--MultiResUNet.pth.tar'
    # save_path = '/home/user/whn/IDRiD/Models/EX/Attention_UNet/save/models/Test_session_04.30 11h05/best_model.WeightedDiceBCE--Attention_UNet.pth.tar'

    tasks = ['EX']
    tensorboard_folder = save_path + 'tensorboard_logs/'

    if tasks[0]=='EX':
        att_vis_path = 'EX/'+model_type + '/vis/Pred/'
    else:

        att_vis_path = model_type + '/vis/Pred/'
    # att_vis_path = model_type + '/vis/fixed_En1/'
    if not os.path.exists(att_vis_path+'org/'):
        os.makedirs(att_vis_path+'org/')



    checkpoint = torch.load(save_path, map_location='cuda')


    if model_type == 'UNet':
        model = UNet()
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        criterion = WeightedDiceBCE()

    elif model_type == 'UNetPP':
        model = smp.UnetPlusPlus(
            encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        criterion = WeightedDiceBCE()

    elif model_type == 'ResUNetPP':
        model = ResUNetPlusPlus(deep_supervision=False)
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        criterion = WeightedDiceBCE()
    elif model_type == 'Attention_UNet':
        model = AttU_Net()
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        criterion = WeightedDiceBCE()


    elif model_type == 'MultiResUNet':
        model = MultiResUnet(in_channels=1, out_channels=1,nf=16)
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        criterion = WeightedDiceBCE()
    elif model_type == 'LBTWDS_MSF_UpDec_ResUNet' or model_type =='LBTWDS_MSF_UpDec_ResUNet_SN':
        model = SN_MSF_UpDec_ResUNet()
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        criterion = LBTW_Loss(WeightedDiceBCE())



    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    print model
    model = model.module

    # Load Datasets
    train_loader, val_loader = load_train_val_data(tasks=tasks, data_path=data_path, batch_size=1, green=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0)  # Choose optimize

    dice_pred = 0.0
    dice_ens = 0.0
    with tqdm(total=27, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        # for i in range(test_data.shape[0]):
        for (i, sample) in enumerate(val_loader, 1):
            test_data, test_label = sample['image'], sample['masks']
            # print "eeeeee",test_data.size()
            arr=test_data.numpy()
            # print "arr",arr.shape
            arr = arr.astype(np.float32())

            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[2], lab.shape[3])) * 255

            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            # 去除图像周围的白边
            height, width = 640, 640
            # 如果dpi=300，那么图像大小=height*width
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(att_vis_path+str(i)+"_lab.jpg", dpi=300)
            plt.close()

            # img_RGB = cv2.imread(att_vis_path+'org/'+str(i)+"_original.jpg", 1)
            # img = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)


            input_img = torch.from_numpy(arr)
            dice_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                           att_vis_path+str(i),
                                                           # layer_name=['preBlock','up_out1_gate0','up_out2_gate0','up_out3_gate0'])
                                                           dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred+=dice_pred_t
            torch.cuda.empty_cache()
            pbar.update()
        print "dice_pred",dice_pred/27.0



