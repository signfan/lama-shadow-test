from mypredict import predictWithDir
from omegaconf import OmegaConf

indir = 'D:/performance' #image, mask
gtdir = 'D:/test_C_fixed_official' # gt_image
#model_path = 'D:/UI/model/pretrained_model' # model
model_path = 'D:/UI/model/shadow_aug_epoch15'
calc_rmse = False
img_suffix = '.jpg'
edge_inpaint = {'use': True, 'kernel_size': 10, 'input': 'predicted_image'}
#edge_inpaint = {'use': True, 'kernel_size': 10, 'input': 'inpainted'}
#edge_inpaint = {'use': False}
edge_inpaint = OmegaConf.create(edge_inpaint)
predictWithDir(indir, gtdir, model_path, calc_rmse, img_suffix, edge_inpaint)
