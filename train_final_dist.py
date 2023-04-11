import os
import sys
import time
import random
import string
import argparse
import re

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from module.Opt2Model import CreateModel
from test_final import validation
from utils import get_args
import utils_dist as utils

import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from module.CustomLoss import MultiCELoss

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')

    val_opt = copy.deepcopy(opt)
    val_opt.eval = True
    
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=val_opt)
    valid_dataset, _ = hierarchical_dataset(root=opt.valid_data, opt=val_opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
        
    """ model configuration """
    converter = TokenLabelConverter(opt)
        
    opt.num_class = len(converter.character)
    print(opt.num_class)
    if opt.rgb:
        opt.input_channel = 3
    if opt.CRM or opt.Predator or opt.Finder:
        model = CreateModel(opt)
    else:
        print('You must specify --CRM or --Predator or --Finder.')
        exit()

    print(model)

    # data parallel for multi-GPU
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
    
    model.train()
    
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'), strict=True)

    """ setup loss """
    if opt.Prediction in ['Attn'] or (not opt.Prediction and opt.Transformer in ['mgp-str','char-str']):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        if opt.MultiCELoss != 0:
            criterion = MultiCELoss(opt.MultiCELoss,opt.batch_max_length+2)
    elif opt.Prediction in ['CTC']:
        criterion = torch.nn.CTCLoss()
    else:
        print('No Prediction Specified.')
        exit()
        
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    scheduler = None
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    if opt.scheduler:
        # scheduler = CosineAnnealingLR(optimizer, T_max=int(opt.num_iter))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.T_0)
    # set up GradScaler for amp
    if opt.amp:
        print('USE AMP')
        scaler = GradScaler()
    """ final options """
    # print(opt)
    with open(f'{opt.saved_path}/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        #print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    iteration = start_iter
    

    print("LR",scheduler.get_last_lr()[0] if opt.scheduler else opt.lr)

        
    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        
        if (opt.Transformer in ["mgp-str"]):
            len_target, char_target = converter.char_encode(labels) 
            bpe_target = converter.bpe_encode(labels)
            wp_target = converter.wp_encode(labels)
            
            char_preds, bpe_preds, wp_preds = model(image)
            
            char_loss = criterion(char_preds.view(-1, char_preds.shape[-1]), char_target.contiguous().view(-1))
            bpe_pred_cost = criterion(bpe_preds.view(-1, bpe_preds.shape[-1]), bpe_target.contiguous().view(-1)) 
            wp_pred_cost = criterion(wp_preds.view(-1, wp_preds.shape[-1]), wp_target.contiguous().view(-1)) 
            cost = char_loss + bpe_pred_cost + wp_pred_cost 

        elif (opt.Transformer in ["char-str"] or opt.Prediction in ['Attn']):
            len_target, char_target = converter.char_encode(labels)

            if opt.amp:
                with autocast():
                    char_preds = model(image)[0]
                    if opt.MultiCELoss != 0:
                        char_loss = criterion(char_preds,char_target)   
                        M = opt.MultiCELoss #3
                        T = opt.batch_max_length+2 #27
                        char_preds = char_preds[:,(M-1)*T:M*T,:]       
                    else:
                        char_loss = criterion(char_preds.view(-1, char_preds.shape[-1]), char_target.contiguous().view(-1))
                    cost = char_loss
            else:
                char_preds = model(image)[0]
                if opt.MultiCELoss != 0:
                    char_loss = criterion(char_preds,char_target)   
                    M = opt.MultiCELoss #3
                    T = opt.batch_max_length+2 #27
                    char_preds = char_preds[:,(M-1)*T:M*T,:]      
                else:
                    char_loss = criterion(char_preds.view(-1, char_preds.shape[-1]), char_target.contiguous().view(-1))
                cost = char_loss

        model.zero_grad()
        if opt.amp:
            #print('I am amp, I am scaling.')
            scaler.scale(cost).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

        loss_avg.add(cost)

        # validation part
        if utils.is_main_process() and ((iteration + 1) % opt.valInterval == 0 or iteration == 0): # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            print("LR",scheduler.get_last_lr()[0] if opt.scheduler else opt.lr)
            with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, current_accuracys, char_preds, confidence_score, labels, infer_time, length_of_data, _ = validation(
                        model, criterion, valid_loader, converter, opt)
                    char_accuracy = current_accuracys[0]
                    bpe_accuracy = current_accuracys[1]
                    wp_accuracy = current_accuracys[2]
                    final_accuracy = current_accuracys[3]
                    cur_best = max(char_accuracy, bpe_accuracy, wp_accuracy, final_accuracy)
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] LR: {scheduler.get_last_lr()[0] if opt.scheduler else opt.lr}, Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"char_accuracy":17s}: {char_accuracy:0.3f}, {"bpe_accuracy":17s}: {bpe_accuracy:0.3f}, {"wp_accuracy":17s}: {wp_accuracy:0.3f}, {"fused_accuracy":17s}: {final_accuracy:0.3f}'

                # keep best accuracy model (on valid dataset)
                if cur_best > best_accuracy:
                    best_accuracy = cur_best
                    torch.save(model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/best_accuracy.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], char_preds[:5], confidence_score[:5]):
                    if opt.Transformer:
                        pred = pred[:pred.find('[s]')]
                    elif 'Attn' in opt.Prediction:
                        pred = pred[:pred.find('[s]')]
                    elif 'CTC' in opt.Prediction:
                        pred = pred.replace('[GO]','')
                        gt = gt.replace('[GO]','')

                    # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                    if opt.sensitive and opt.data_filtering_off:
                        pred = pred.lower()
                        gt = gt.lower()
                        alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                        out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                        pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                        gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if utils.is_main_process() and (iteration + 1) % 5e+3 == 0:
            torch.save(
                model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
        #if utils.is_main_process():
        #    print(f'{iteration}/{opt.num_iter},loss: {cost}')
        if scheduler is not None:
            scheduler.step()

if __name__ == '__main__':

    opt = get_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.TransformerModel}' if opt.Transformer else f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'{opt.saved_path}/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
    #if True:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    
    utils.init_distributed_mode(opt)

    print(opt)
    
    """ Seed and GPU setting """
    
    seed = opt.manualSeed + utils.get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    train(opt)

