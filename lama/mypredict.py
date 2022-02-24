import os
##os.chdir('lama')
##os.environ['TORCH_HOME'] = os.getcwd()
##os.environ['PYTHONPATH'] = os.getcwd()

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from calc_rmse_eval import main

from saicinpainting.training.trainers import load_checkpoint
from torch.utils.data._utils.collate import default_collate
from saicinpainting.training.data.datasets import make_default_val_dataset

def move_to_device(obj, device):
    return {name: val.to(device) for name, val in obj.items()}

def predict(data, model_path):
    device = torch.device('cuda')
    # print(model_path)
    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
        
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(model_path, 'best.ckpt')
    # print('before model')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    # print(model)
    # print('a')
    with torch.no_grad():
        # print('b')
        # print(data)
        tmp = default_collate([data])
        # print('collated: ', tmp)
        batch = move_to_device(tmp, device)
        # print('device: ', batch)
        batch['mask'] = (batch['mask'] > 0) * 1
        # print('batch: ', batch)
        # print('ok')
        # print(batch['image'].shape, batch['mask'].shape)
        batch = model(batch)
        # print('modeled: ', batch)
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        #cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cur_res2 = batch['predicted_image'][0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res2 = np.clip(cur_res2 * 255, 0, 255).astype('uint8')
        if 'shape' in batch.keys():
            height, width = torch.squeeze(batch['shape'])
            # print(cur_res.shape)
            cur_res = cur_res[:height, :width, :]
            cur_res2 = cur_res2[:height, :width, :]
        #cur_res2 = cv2.cvtColor(cur_res2, cv2.COLOR_RGB2BGR)
    return cur_res, cur_res2, model

def predictWithModel(data, m):
    device = torch.device('cuda')

    model = m
    model.freeze()
    model.to(device)
    # print(model)
    # print('a')
    with torch.no_grad():
        # print('b')
        # print(data)
        tmp = default_collate([data])
        # print('collated: ', tmp)
        batch = move_to_device(tmp, device)
        # print('device: ', batch)
        batch['mask'] = (batch['mask'] > 0) * 1
        # print('batch: ', batch)
        batch = model(batch)
        # print('modeled: ', batch)
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        #cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cur_res2 = batch['predicted_image'][0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res2 = np.clip(cur_res2 * 255, 0, 255).astype('uint8')
        if 'shape' in batch.keys():
            height, width = torch.squeeze(batch['shape'])
            # print(cur_res.shape)
            cur_res = cur_res[:height, :width, :]
            cur_res2 = cur_res2[:height, :width, :]
        #cur_res2 = cv2.cvtColor(cur_res2, cv2.COLOR_RGB2BGR)
    return cur_res, cur_res2

def predictWithDir(indir, gtdir, model_path, calc_rmse, img_suffix, edge_inpaint):
    device = torch.device('cuda')
    print(model_path)
    train_config_path = os.path.join(model_path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
        
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    predict_config = dict({'kind': 'default', 'img_suffix': img_suffix, 'pad_out_to_modulo': 8})
    predict_config = OmegaConf.create(predict_config)

    checkpoint_path = os.path.join(model_path, 'best.ckpt')
    print('before model')
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    print(model)
    print('a')

    if 'hsv' in os.path.basename(model_path):
        predict_config.kind = 'shadow_HSV'
    dataset = make_default_val_dataset(indir=indir, **predict_config)
    subdir = f'{os.path.basename(model_path)}_{os.path.basename(indir)}'
    if edge_inpaint.use:
        subdir += f'_edge{edge_inpaint.kernel_size}_{edge_inpaint.input}'
    outdir = os.path.abspath(os.path.join(model_path, os.pardir, os.pardir, 'output', subdir, 'inpainted'))
    outdir2 = os.path.abspath(os.path.join(model_path, os.pardir, os.pardir, 'output', subdir, 'predicted'))

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(outdir2):
        os.makedirs(outdir2)

    if edge_inpaint.use:
        model_path = 'D:/UI/model/pretrained_model'
        train_config_path = os.path.join(model_path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(model_path, 'best.ckpt')
        model2 = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model2.freeze()
        model2.to(device)

    fnames = dataset.img_filenames
    with torch.no_grad():
        for i, data in enumerate(dataset):
            # print('b')
            tmp = default_collate([data])
            # print('collated: ', tmp)
            batch = move_to_device(tmp, device)
            # print('device: ', batch)
            batch['mask'] = (batch['mask'] > 0) * 1
            # print('batch: ', batch)
            # print('ok')
            # print(batch['image'].shape, batch['mask'].shape)
            batch = model(batch)

            if edge_inpaint.use:
                tmp = dict()
                tmp['image'] = batch[edge_inpaint.input]
                kernel_size = edge_inpaint.kernel_size
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                src = batch['mask'].detach().cpu().numpy()[0][0].astype('uint8')
                dst = cv2.dilate(src, kernel, iterations=1) - cv2.erode(src, kernel, iterations=1)
                dst = np.expand_dims(np.expand_dims(dst, 0), 0)
                tmp['mask'] = torch.tensor(dst).to(device)
                tmp['shape'] = batch['shape']
                batch = model2(tmp)

            cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res2 = batch['predicted_image'][0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res2 = np.clip(cur_res2 * 255, 0, 255).astype('uint8')

            if 'shape' in batch.keys():
                height, width = torch.squeeze(batch['shape'])
                cur_res = cur_res[:height, :width, :]
                cur_res2 = cur_res2[:height, :width, :]

            fname = os.path.join(outdir, os.path.basename(fnames[i]))
            fname2 = os.path.join(outdir2, os.path.basename(fnames[i]))
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cur_res2 = cv2.cvtColor(cur_res2, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fname, cur_res)
            cv2.imwrite(fname2, cur_res2)
    if calc_rmse:
        print('Calc RMSE')
        main(indir, gtdir, outdir, True)
        main(indir, gtdir, outdir2, True)
    return #cur_res, cur_res2, model