from PIL import Image, ImageChops
import math
import numpy as np
import numpy.ma as ma
from skimage import io, color
import cv2
from skimage.color  import rgb2lab
import torch
import torchvision
import torch.nn.functional as F
from typing import Dict, Iterable, Callable, List, Tuple
import argparse
from prettytable import PrettyTable
import sys
import os
from tqdm import tqdm
import time    
from metrics import ber, accuracy
###################################

def rgb_to_lab_space(img):
    to_squeeze = (img.dim() == 3)
    device = img.device

    img = img.detach().cpu()
    if to_squeeze:
        img = img.unsqueeze(0)

    img = img.permute(0, 2, 3, 1).numpy()
    transformed = rgb2lab(img)
    output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
    if to_squeeze:
        output = output.squeeze(0)
    
    return output.to(device)

def rmse_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    loss = torch.sqrt(F.mse_loss(output, target, reduction = 'mean'))
    mae = F.l1_loss(output, target, reduction = 'mean') # Mean Absolute Loss
    return loss, mae

def rmse_loss_shadow(output_free: Tuple[torch.Tensor], target_mask: Tuple[torch.Tensor], target_free: Tuple[torch.Tensor]) -> float:
    #output_free = output#[0].detach().clone()
    #target_free = target#[0].detach().clone()
    #target_mask = target_mask#[1].detach()
    shadow_part = (target_mask > 0.5).repeat(3, 1, 1)
    ret = (output_free - target_free) ** 2
    return math.sqrt(ret[shadow_part].mean().item())


def rmse_loss_non_shadow(output_free: Tuple[torch.Tensor], target_mask: Tuple[torch.Tensor], target_free: Tuple[torch.Tensor]) -> float:
    #output_free = output#[0].detach().clone()
    #target_free = target#[0].detach().clone()
    #target_mask = target_mask#[1].detach()

    non_shadow_part = (target_mask < 0.5).repeat(3, 1, 1)
    ret = (output_free - target_free) ** 2
    return math.sqrt(ret[non_shadow_part].mean().item())

#---------------------------------------------------------
# Numpy Part
#---------------------------------------------------------
def rmse_loss_np(output: np, target: np) -> float:
    loss = np.sqrt(((output - target) ** 2).mean())
    return loss

def rmse_loss_shadow_np(output_free: np, target_mask: np, target_free: np) -> float:
    #output_free = output
    #target_free = target
    #target_mask = target_mask
    target_mask = target_mask > 0.5
    shadow_part = np.tile(target_mask, (3,1,1))
    ret = (output_free - target_free) ** 2
    return math.sqrt(ret[shadow_part].mean())

def rmse_loss_non_shadow_np(output_free: np, target_mask: np, target_free: np) -> float:
    #output_free = output
    #target_free = target
    #target_mask = target_mask
    target_mask = target_mask < 0.5
    shadow_part = np.tile(target_mask, (3,1,1))
    ret = (output_free - target_free) ** 2
    return math.sqrt(ret[shadow_part].mean())



def calc_accuracy_by_tensor(opt, img_list):
#    ber(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    accuracy_sum_overall = 0.0
    simple_accuracy_sum_overall = 0.0
    ###################################
    #for idx, img_name in enumerate(tqdm(img_list)):
    for idx, img_name in enumerate(img_list):        
        predict_mask = Image.open(os.path.join(opt.predict_maskroot, img_name + opt.img_format)).convert('L')
        mask = Image.open(os.path.join(opt.maskroot, img_name + opt.img_format)).convert('L')

        predict_mask  = torchvision.transforms.ToTensor()(predict_mask)
        mask    = torchvision.transforms.ToTensor()(mask)
        accuracy_simple = accuracy(predict_mask, mask)        
        simple_accuracy_sum_overall += accuracy_simple
        
    avg_accuray_overall = accuracy_sum_overall / len(img_list)
    
    print(f"Accuracy WholeImage : {avg_accuray_overall} = {accuracy_sum_overall} / {len(img_list)}")

                

def calc_ber_by_tensor(opt, img_list):
#    ber(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    ber_sum_shadow = 0.0
    ber_sum_non_shadow = 0.0
    ber_sum_overall = 0.0

    accuracy_sum_overall = 0.0
    simple_accuracy_sum_overall = 0.0

    ###################################
    #for idx, img_name in enumerate(tqdm(img_list)):
    for idx, img_name in enumerate(img_list):        
        predict_mask = Image.open(os.path.join(opt.predict_maskroot, img_name + opt.img_format)).convert('L')
        mask = Image.open(os.path.join(opt.maskroot, img_name + opt.img_format)).convert('L')

        predict_mask  = torchvision.transforms.ToTensor()(predict_mask)
        mask    = torchvision.transforms.ToTensor()(mask)
        ber_overall = ber(predict_mask, mask)
        ber_sum_overall += ber_overall

        accuracy_simple = accuracy(predict_mask, mask)        
        accuracy_sum_overall += accuracy_simple

        print(f"processing... {idx} / {len(img_list)} BER: {ber_overall:10.6f}, Accuracy: 5{accuracy_simple:10.6f}")        
        if opt.file_mask_out != None:    
            opt.file_mask_out.write(f"{ber_overall:10.6f}, {accuracy_simple:10.6f}\n")

    print("--Resule : Mask----------------")
    avg_accuray_overall = accuracy_sum_overall / len(img_list)
    print(f">> Accuracy WholeImage : {avg_accuray_overall} = {accuracy_sum_overall} / {len(img_list)}")
    avg_ber_overall = ber_sum_overall / len(img_list)
    print(f">> BER Overwhole : {avg_ber_overall} = {ber_sum_overall} / {len(img_list)}")
                

def calc_rmse_by_tensor(opt, img_list):
    
    rmse_sum_shadow = 0.0
    rmse_sum_non_shadow = 0.0
    rmse_sum_overall = 0.0

    mae_sum_overall = 0.0
    

    ###################################
    #for idx, img_name in enumerate(tqdm(img_list)):
    for idx, img_name in enumerate(img_list):        
        #print('processing... {} / {}'.format(idx+1, len(img_list)))

        target = Image.open(os.path.join(opt.targetroot, img_name + opt.img_format)).convert('RGB')#.resize((320,240))   # resize((width, height))
        gt = Image.open(os.path.join(opt.gtroot, img_name + opt.img_format)).convert('RGB')#.resize((320,240))
        if opt.use_mask == True:
            # shk: mask name change
            mask = Image.open(os.path.join(opt.maskroot, img_name + '_mask' + opt.img_format)).convert('L')#.resize((320,240))
        
        if opt.resize_256 == True:
            target  = target.resize((256,256))
            gt      = gt.resize((256, 256))
            if opt.use_mask == True:
                mask    = mask.resize((256, 256))
        
        if idx == 0:
            print(f"idx: {idx}, Image Size: {target.size}, {gt.size}")
        
        target  = torchvision.transforms.ToTensor()(target)
        gt      = torchvision.transforms.ToTensor()(gt)
        if opt.use_mask == True:
            mask    = torchvision.transforms.ToTensor()(mask)
        target  = rgb_to_lab_space(target)
        gt  = rgb_to_lab_space(gt)
        
        rmse_overall, mae_overall = rmse_loss(target, gt)
        if opt.use_mask == True:
            rmse_shadow = rmse_loss_shadow(target, mask, gt)
            rmse_non_shadow = rmse_loss_non_shadow(target, mask, gt)
            
        rmse_sum_overall += rmse_overall
        if opt.use_mask == True:
            rmse_sum_shadow += rmse_shadow
            rmse_sum_non_shadow += rmse_non_shadow

        if opt.file_sdf_out != None:    
            opt.file_sdf_out.write(f"{rmse_overall:10.6f},{rmse_shadow:10.6f},{rmse_non_shadow:10.6f}\n")
        mae_sum_overall   += mae_overall
        
        print('processing... {} / {}, overall: {:10.6f}, shadow: {:10.6f}, non_shadow: {:10.6f}, [{}]'.format(idx+1, len(img_list), rmse_overall, rmse_shadow, rmse_non_shadow, img_name))

    avg_rmse_shadow = rmse_sum_shadow / len(img_list)
    avg_rmse_non_shadow = rmse_sum_non_shadow / len(img_list)
    avg_rmse_overall = rmse_sum_overall / len(img_list)
    
    avg_mae_overall = mae_sum_overall / len(img_list)

    # print(f"-- Result : RMSE ---------------")
    # print(f">> shadow: \t{avg_rmse_shadow:10.6f} \t= {rmse_sum_shadow:15.5} / {len(img_list)}")
    # print(f">> non_shadow:\t{avg_rmse_non_shadow:10.6f}\t= {rmse_sum_non_shadow:15.5} / {len(img_list)}")
    # print(f">> overall:\t{avg_rmse_overall:10.6f} \t= {rmse_sum_overall:15.5} / {len(img_list)}")
    # print(f"-- Result : MAE (Mean Absolute Error ---------------")
    # print(f">> overall:\t{avg_mae_overall:10.6f} \t= {mae_sum_overall:15.5} / {len(img_list)}")
    log = (
        f"-- Result : RMSE ---------------" + '\n' +
        f">> shadow: \t{avg_rmse_shadow:10.6f} \t= {rmse_sum_shadow:15.5} / {len(img_list)}" + '\n' +
        f">> non_shadow:\t{avg_rmse_non_shadow:10.6f}\t= {rmse_sum_non_shadow:15.5} / {len(img_list)}" + '\n' +
        f">> overall:\t{avg_rmse_overall:10.6f} \t= {rmse_sum_overall:15.5} / {len(img_list)}" + '\n' +
        f"-- Result : MAE (Mean Absolute Error ---------------" + '\n' +
        f">> overall:\t{avg_mae_overall:10.6f} \t= {mae_sum_overall:15.5} / {len(img_list)}"
    )
    print(log)
    opt.file_sdf_out.write(log)


def calc_rmse_by_numpy(opt, img_list):
    
    rmse_sum_shadow = 0.0
    rmse_sum_non_shadow = 0.0
    rmse_sum_overall = 0.0

    ###################################
    #for idx, img_name in enumerate(tqdm(img_list)):
    for idx, img_name in enumerate(img_list):    
        #print('processing... {} / {}'.format(idx+1, len(img_list)))

        target = Image.open(os.path.join(opt.targetroot, img_name + opt.img_format)).convert('RGB')#.resize((320,240))   # resize((width, height))
        gt = Image.open(os.path.join(opt.gtroot, img_name + opt.img_format)).convert('RGB')#.resize((320,240))
        mask = Image.open(os.path.join(opt.maskroot, img_name + opt.img_format)).convert('L')#.resize((320,240))
        
        target  = np.array(target)
        gt      = np.array(gt)
        mask    = np.array(mask)
        target  = color.rgb2lab(target).transpose(2, 0, 1)
        gt      = color.rgb2lab(gt).transpose(2, 0, 1)
        rmse_overall = rmse_loss_np(target, gt)
        rmse_shadow = rmse_loss_shadow_np(target, mask, gt)
        rmse_non_shadow = rmse_loss_non_shadow_np(target, mask, gt)
        
        rmse_sum_shadow += rmse_shadow
        rmse_sum_non_shadow += rmse_non_shadow
        rmse_sum_overall += rmse_overall
        #print('processing... {} / {}, overall: {:10.6f}, shadow: {:10.6f}, non_shadow: {:10.6f}'.format(idx+1, len(img_list), rmse_overall, rmse_shadow, rmse_non_shadow))
        print('processing... {} / {}, overall: {:10.6f}, shadow: {:10.6f}, non_shadow: {:10.6f}, [{}]'.format(idx+1, len(img_list), rmse_overall, rmse_shadow, rmse_non_shadow, img_name))
    avg_rmse_shadow = rmse_sum_shadow / len(img_list)
    avg_rmse_non_shadow = rmse_sum_non_shadow / len(img_list)
    avg_rmse_overall = rmse_sum_overall / len(img_list)

    print(f"-- Result : RMSE ---------------")
    print(f">> shadow: \t{avg_rmse_shadow:10.6f} \t= {rmse_sum_shadow:15.5} / {len(img_list)}")
    print(f">> non_shadow:\t{avg_rmse_non_shadow:10.6f}\t= {rmse_sum_non_shadow:15.5} / {len(img_list)}")
    print(f">> overall:\t{avg_rmse_overall:10.6f} \t= {rmse_sum_overall:15.5} / {len(img_list)}")


def print_option(opt, column = 1):

    if column == 1:
        table = PrettyTable(["key", "Argument"])
    else:
        table = PrettyTable(["key1", "value1", "key2", "value2"])

    table.title = "Calculate Shadow Removal Performance"
    table.align = "l"
    
    config_iter = iter(opt.__dict__.items())
    while True:
        try:
            key1 = next(config_iter)
            if column == 1:
                table.add_row([key1[0], key1[1]]) #, fill (value,width=50]])                        
            print(f"{key1[0]:20}:{key1[1]}")#, end='\t\t')    
        except StopIteration:
            break
        if column == 2:
            try:
                key2 = next(config_iter)
                #print(f"{key[0]:20}:{key[1]}", end='\n')    
                table.add_row([key1[0], key1[1], key2[0], key2[1]]) #, fill (value,width=50]])
                #table.add_row([key2[0], key2[1]]) #, fill (value,width=50]])
            except StopIteration:
                break
    print(table)
    


#if __name__ == '__main__':
def main(indir, gtdir, outdir, use_mask):
    print("_________________________________________________________________")
    parser = argparse.ArgumentParser()
    parser.add_argument('--targetroot', type=str, default='/home/juwan/Projects/pytorch_framework/runs/general/FINAL_Dual_Skip_mse_for_fast_compare_ISTD/210924-0829.BATCH2.480x640Resume.ISTD+/epoch_250.chk.ISTD_Dataset.480X640TestSDFResults', help='Predict Shadow Free File Path')
    parser.add_argument('--gtroot', type=str, default='/data2/SharedData/ISTD_Dataset/test/test_C_fixed_official/', help='Ground Truth Shadow Free File Path')
    
    parser.add_argument('--predict_maskroot', type=str, default='/home/juwan/Projects/pytorch_framework/runs/general/FINAL_Dual_Skip_mse_for_fast_compare_ISTD/210924-0829.BATCH2.480x640Resume.ISTD+/epoch_250.chk.ISTD_Dataset.480X640TestMaskResults', help='Predict Mask File Path')
    parser.add_argument('--maskroot', type=str, default='/data2/SharedData/ISTD_Dataset/test/test_B/', help='Ground Truth Mask File Path')
    parser.add_argument('--img_format', type=str, default='.png', help='image format')
    parser.add_argument('--use_tensor', type=bool, default=True, help='calculation method: Numpy or Tensor')
    parser.add_argument('--resize_256', type=bool, default=False, help='Image Resize (No(default) / Yes) ')
    parser.add_argument('--use_mask', type=bool, default=False, help='Use Shadow Mask Area')
    parser.add_argument('--write_metric', type=str, default='metric.txt', help='metric filename')
    opt = parser.parse_args()
    
    #opt.targetroot = '/data2/SharedData/ISTD_Dataset/test/test_A/'
    # opt.targetroot = '/data2/SharedData/2021.SDR_PROJECT_TEST/FINAL_Dual_Skip_mse_for_fast_compare_ISTD.epoch_075_3.218'
    # opt.targetroot = '/data2/SharedData/2021.SDR_PROJECT_TEST/FINAL_Dual_Skip_mse_for_fast_compare_ISTD.epoch_025_2.955'
    
    #opt.targetroot = '/home/juwan/Projects/pytorch_framework/runs/general/FINAL_Dual_Skip_mse_for_fast_compare_ISTD/210924-0829.BATCH2.480x640Resume.ISTD+/epoch_250.chk.ISTD_Dataset.480X640TestSDFResults'            
    # opt.predict_maskroot = '/home/juwan/Projects/pytorch_framework/runs/general/FINAL_Dual_Skip_mse_for_fast_compare_ISTD/210924-0829.BATCH2.480x640Resume.ISTD+/epoch_250.chk.ISTD_Dataset.480X640TestMaskResults'
    #opt.predict_maskroot = '/data2/SharedData/ISTD_Dataset/test/test_A/'
    # '''        
    # opt.targetroot = '/home/juwan/Projects/pytorch_framework/runs/general/FINAL_Dual_Skip_mse_for_fast_compare_ISTD/210924-0829.BATCH2.480x640Resume.ISTD+/epoch_250.chk.ISTD_Dataset.480X640TestSDFResults'
    # opt.gtroot = '/data2/SharedData/ISTD_Dataset/test/test_C_fixed_official/'
    # opt.maskroot = '/data2/SharedData/ISTD_Dataset/test/test_B/'
    # '''
    opt.targetroot = outdir
    opt.gtroot = gtdir
    opt.maskroot = indir
    opt.use_mask = use_mask
    
    img_list = [os.path.splitext(f)[0] for f in sorted(os.listdir(opt.targetroot)) if f.upper().endswith(".JPG") or f.upper().endswith(".PNG")]



    #img_list = img_list[0:10]
    
    opt.file_count = len(img_list)

    print_option(opt, 1)



    opt.file_sdf_out = None
    # output file
    if opt.write_metric != None:
        output_sdf_path = os.path.join(opt.targetroot, opt.write_metric)
        opt.file_sdf_out = open(output_sdf_path, 'w')
        opt.file_sdf_out.write("rmse_overall, rmse_shadow, rmse_non_shadow\n")

        # output_mask_path = os.path.join(opt.predict_maskroot, opt.write_metric)
        # opt.file_mask_out = open(output_mask_path, 'w')
        # opt.file_mask_out.write("BER, Accuracy\n")
    

    
    if True: #opt.use_tensor == True:
        print(f"calc_rmse_by_tensor---------------------")        
        s_time = time.time()
        calc_rmse_by_tensor(opt, img_list)
        print("Tensor---{}s seconds---".format(time.time()-s_time))
    '''
    if True: #opt.use_tensor == False:
        print(f"calc_rmse_by_numpy---------------------")        
        s_time = time.time()
        calc_rmse_by_numpy(opt, img_list)
        print("Numpy---{}s seconds---".format(time.time()-s_time))
    '''
    #calc_ber_by_tensor(opt, img_list)        
    #calc_accuracy_by_tensor(opt, img_list)

    if opt.file_sdf_out != None:
        opt.file_sdf_out.close()
    # if opt.file_mask_out != None:        
    #     opt.file_mask_out.close()

    print(f"Target Data: {opt.targetroot}")
    print(f"Ground Data: {opt.gtroot}")
    print(f"256 Resized: {opt.resize_256}")
    print(f"FileCount: {opt.file_count}")

    