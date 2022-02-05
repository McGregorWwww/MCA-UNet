# coding=utf-8
import tensorflow as tf
import torch
import torch.optim
from tensorboardX import SummaryWriter
import os
import time
import numpy as np
from torch.autograd import Variable
from utils import auc_on_batch, aupr_on_batch, plot, to_var, to_numpy
from DataSet import load_train_val_data
from nets.CAUNet import CAUNet
from nets.msf_updec_sum_net import *
import torch.nn as nn
from Losses import *
from LR_methods import CosineAnnealingWarmRestarts
import logging
import warnings
warnings.filterwarnings("ignore")

## PARAMETERS OF THE MODEL
use_cuda = torch.cuda.is_available()
learning_rate = 5e-3
# image_size = (512, 512)
n_labels = 1
epochs = 300
batch_size = 4
print_frequency = 1
save_frequency = 10
save_model = True
# tumor_percentage = 0.5
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def logger_config(log_path):
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    loggerr = logging.getLogger()
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    loggerr.setLevel(level=logging.INFO)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    # formatter = logging.Formatter('%(asctime)s -%(levelname)s:%(message)s')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 为logger对象添加句柄
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, auc, average_auc, aupr, average_aupr, mode, lr):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += '{}: {:.3f} '.format(loss_name, loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'AUC {:.3f} '.format(auc)
    string += '(Avg {:.4f}) '.format(average_auc)
    string += 'AUPR {:.4f} '.format(aupr)
    string += '(Avg {:.4f}) '.format(average_aupr)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    string += 'Time {:.1f} '.format(batch_time)
    string += '(Avg {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    val_loss = state['val_loss']  # loss value
    model = state['model']  # model type
    loss = state['loss']  # loss name

    if best_model:
        filename = save_path + '/' + \
                   'best_model.{}--{}.pth.tar'.format(loss, model)
    else:
        filename = save_path + '/' + \
                   'model.{}--{}--{:02d}.pth.tar'.format(loss, model, epoch)

    torch.save(state, filename)


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, lbtw_algorithm,lbtw_algorithm_four):
    logging_mode = 'Train' if model.training else 'Val'


    end = time.time()
    time_sum, loss_sum = 0, 0
    train_aupr, train_auc = 0.0, 0.0  # train_auc is the train area under the ROC curve
    aupr_sum, auc_sum = 0.0, 0.0

    auprs = []
    for (i, sample) in enumerate(loader, 1):

        images, masks = sample['image'], sample['masks']
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images = Variable(images.cuda(async=True))
        masks = Variable(masks.cuda(async=True))
        images = images.float()
        masks = masks.float()

        # ====================================================
        #             Compute loss
        # ====================================================


        output, pred_comb2, pred_comb3, pred_comb1 = model(images)
        losses, out_loss, c3_loss, c2_loss, c1_loss = criterion(output, pred_comb3, pred_comb2, pred_comb1, masks)  # Loss
        if model.training:
            loss, w0, w3, w2, w1 = lbtw_algorithm(i, out_loss, c3_loss, c2_loss, c1_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_auc = auc_on_batch(masks, output)
        train_aupr = aupr_on_batch(masks, output)
        auprs.append(train_aupr)

        # measure elapsed time
        batch_time = time.time() - end

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        auc_sum += len(images) * train_auc
        aupr_sum += len(images) * train_aupr

        if i == len(loader):
            average_loss = loss_sum / (batch_size*(i-1) + len(images))
            average_time = time_sum / (batch_size*(i-1) + len(images))
            train_auc_average = auc_sum / (batch_size*(i-1) + len(images))
            train_aupr_avg = aupr_sum / (batch_size*(i-1) + len(images))


        else:
            average_loss = loss_sum / (i * batch_size)
            average_time = time_sum / (i * batch_size)
            train_auc_average = auc_sum / (i * batch_size)
            train_aupr_avg = aupr_sum / (i * batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_auc, train_auc_average, train_aupr, train_aupr_avg, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups))

        if tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_auc', train_auc, step)
            writer.add_scalar(logging_mode + '_aupr', train_aupr, step)

        torch.cuda.empty_cache()


    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_aupr_avg






##################################################################################
#=================================================================================
#          Main Loop
#=================================================================================
##################################################################################

def main_loop(data_path, batch_size=batch_size, model_type='', green=False, tensorboard=True, task_name=None):
    # Load train and val data
    tasks = task_name
    # data_path = data_path
    n_labels = len(tasks)
    n_channels = 1 if green else 3  # green or RGB
    train_loader, val_loader = load_train_val_data(tasks=tasks, data_path=data_path, batch_size=batch_size, green=green)


    if model_type == 'CAUNet':
        lr = learning_rate
        model = SN_MSF_UpDec_ResUNet()
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print "Let's use {0} GPUs!".format(torch.cuda.device_count())
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        criterion = LBTW_Loss(WeightedDiceBCE())


    else:
        raise TypeError('Please enter a valid name for the model type')

    try:
        loss_name = criterion._get_name()
    except AttributeError:
        loss_name = criterion.__name__


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Choose optimize
    optimizer = nn.DataParallel(optimizer, device_ids=[0,1,2,3])
    optimizer = optimizer.module
    # lr_scheduler =  None
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=20,min_lr=5e-5, mode='max', factor=0.8)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)
    lbtw_algorithm = LBTW_algorithm()
    lbtw_algorithm_four = LBTW_algorithm_four()
    if tensorboard:
        log_dir = tensorboard_folder + session_name + '/'
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_aupr = 0.0
    best_epoch = 1
    for epoch in range(epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, epochs + 1))
        logger.info(session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, lbtw_algorithm, lbtw_algorithm_four)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_aupr = train_one_epoch(val_loader, model, criterion,
                                            optimizer, writer, epoch, lr_scheduler,model_type,
                                                 lbtw_algorithm, lbtw_algorithm_four)


        # =============================================================
        #       Save best model
        # =============================================================
        if val_aupr > max_aupr and epoch+1 > 5:
            logger.info('\t Saving best model, mean AUPR increased from: {:.4f} to {:.4f}'.format(max_aupr,val_aupr))
            max_aupr = val_aupr
            best_epoch = epoch + 1
            save_checkpoint({'epoch': epoch,
                             'best_model': True,
                             'model': model_type,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'loss': loss_name,
                             'optimizer': optimizer.state_dict()}, model_path)
        elif save_model and (epoch + 1) % save_frequency == 0:
            logger.info('\t Saving model, current mean AUPR: {:.4f}, the best: {:.4f}'.format(val_aupr, max_aupr))
            save_checkpoint({'epoch': epoch,
                             'best_model': False,
                             'model': model_type,
                             'loss': loss_name,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict()}, model_path)
        else:
            logger.info('\t Mean AUPR:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_aupr,max_aupr, best_epoch))

    return model


if __name__ == '__main__':
    ## PARAMS
    ## You should create the following save folder and tensorboard folder if the mkdirs command fails to create a folder

    task_name = 'EX'
    model_name = 'CAUNet'


    main_path = '../Segmentation_dataset/' # folder with test_images folder and test_masks folder (containing 'EX' folder ...)
    sets_path = os.path.join(main_path, 'datasets/')
    csv_path = os.path.join(main_path, 'data/tumor_count.csv')
    data_folder = os.path.join(main_path, 'data/')
    if task_name is 'EX':
        save_path = task_name +'/'+ model_name + '/save/'
    else:
        save_path = model_name + '/save/'
    session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')
    model_path = save_path + 'models/' + session_name + '/'
    tensorboard_folder = save_path + 'tensorboard_logs/'
    logger_path = save_path + 'log_file/'
    if not os.path.isdir(logger_path):
        os.makedirs(logger_path)

    logger_path = save_path + 'log_file/' + session_name + ".log"

    logger = logger_config(log_path=logger_path)

    model = main_loop(data_path=main_path, model_type=model_name, green=True, tensorboard=True, task_name=[task_name])

