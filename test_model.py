from nets.UNet import UNet
from main import train_one_epoch
from nets.msf_updec_sum_net import MSF_UpDec_ResUNet
import torch.nn as nn
from DataSet import load_train_val_data
from tensorboardX import SummaryWriter
import os
import time
import torch
from Losses import *

if __name__ == '__main__':
    ## PARAMS
    model_type = 'LBTWDS_MSF_UpDec_ResUNet'
    data_path = '../Segmentation_dataset/' #path where dataset is stored
    save_path = 'LBTWDS_MSF_UpDec_ResUNet/save/models/Test_session_03.16 20h36/best_model.LBTW_WeightedDiceBCE--LBTWDS_MSF_UpDec_ResUNet.pth.tar' #path where weights of our model was downloaded
    tensorboard_folder = save_path + 'tensorboard_logs/'

    session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')
    tasks = ['MA']

    # Tensorboard logs
    log_dir = tensorboard_folder + session_name + '/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # Loading a model
    checkpoint = torch.load(save_path, map_location='cuda')


    if model_type == 'UNet':
        model = UNet(1, 1)
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        # Choose loss function
        criterion = nn.MSELoss()
        # criterion = dice_loss
        # criterion = mean_dice_loss
        # criterion = nn.BCELoss()


    elif model_type == 'LBTWDS_MSF_UpDec_ResUNet':
        model = MSF_UpDec_ResUNet()
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        # Choose loss function
        # criterion = LBTW_Loss(nn.MSELoss())
        # criterion = LBTW_Loss(nn.BCELoss())
        criterion = LBTW_Loss(WeightedDiceBCE())
        # criterion = dice_loss
        # criterion = mean_dice_loss
        # criterion = nn.BCELoss()


    # load weights into model
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')

    # Load Datasets
    train_loader, val_loader = load_train_val_data(tasks=tasks, data_path=data_path, batch_size=4, green=True)
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0)  # Choose optimize
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=7)

    with torch.no_grad():
        model.eval()
        print('Running Evaluation...')
        val_loss, val_aupr = train_one_epoch(val_loader, model, criterion,
                                             optimizer, writer, 0,
                                             model_type=model_type)
        print('AUPR evaluated on validation set:', val_aupr)
