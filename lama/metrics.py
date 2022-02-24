import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.color  import rgb2lab
from typing import Dict, Iterable, Callable, List, Tuple
import math

def test_binary(var1: int, var2: int) -> int:
    return var1 + var2

def test_unary(var1: int, dummy: int) -> int:
    return var1 + var1

def mse_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    loss = F.mse_loss(output, target, reduction = 'mean') 
    return loss

def rmse_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    loss = torch.sqrt(F.mse_loss(output, target, reduction = 'mean'))
    return loss

def rmse_loss_shadow(output: Tuple[torch.Tensor], target: Tuple[torch.Tensor]) -> float:
    output_free = output[0].detach().clone()
    target_free = target[0].detach().clone()
    target_mask = target[1].detach()

    shadow_part = (target_mask > 0.5).repeat(1, 3, 1, 1)
    ret = (output_free - target_free) ** 2
    return math.sqrt(ret[shadow_part].mean().item())


def rmse_loss_non_shadow(output: Tuple[torch.Tensor], target: Tuple[torch.Tensor]) -> float:
    output_free = output[0].detach().clone()
    target_free = target[0].detach().clone()
    target_mask = target[1].detach()

    non_shadow_part = (target_mask < 0.5).repeat(1, 3, 1, 1)
    ret = (output_free - target_free) ** 2
    return math.sqrt(ret[non_shadow_part].mean().item())


def bce_logit_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.binary_cross_entropy_with_logits(output, target)).item()


def bce_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.binary_cross_entropy(output, target)).item()


def cross_entropy_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(F.cross_entropy(output, target)).item()

# output : predicted , target : ground truth
def ber(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold      # ground truth
    output_flat = output.view(batch_size, -1) >= threshold      # predicted mask
    true = target_flat              
    false = torch.logical_not(target_flat)
    positive = output_flat
    negative = torch.logical_not(output_flat)
    TP = torch.sum(torch.as_tensor(torch.logical_and(true, positive), dtype=torch.float), dim=1)
    TN = torch.sum(torch.as_tensor(torch.logical_and(false, negative), dtype=torch.float), dim=1)
    FP = torch.sum(torch.as_tensor(torch.logical_and(false, positive), dtype=torch.float), dim=1)
    FN = torch.sum(torch.as_tensor(torch.logical_and(true, negative), dtype=torch.float), dim=1)
    try:
        tp_rate = TP / (TP + FN)    # 그림자 영역이라고 찾은 것 중에 그림자 비율
        tn_rate = TN / (FP + TN)    # 비 그림자 영역이라고 찾은 것 중에 비 그림자 비율
    except Exception as err:
        print("ber(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:\n")
        print("Error ", err)
        print(f"TN : {TN}, FP : {FP}, FN: {FN}, TP: {TP}")
        exit(0)

    tp_rate[tp_rate != tp_rate] = .0
    tn_rate[tn_rate != tn_rate] = .0

    return 1.0 - torch.mean(.5 * (tp_rate + tn_rate)).item()

#   ber 함수와 동일함...
def ber_old(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold      # ground truth
    output_flat = output.view(batch_size, -1) >= threshold      # predicted mask
    true = target_flat              
    false = torch.logical_not(target_flat)
    positive = output_flat
    negative = torch.logical_not(output_flat)
    TP = torch.sum(torch.as_tensor(torch.logical_and(true, positive), dtype=torch.float), dim=1)
    TN = torch.sum(torch.as_tensor(torch.logical_and(false, negative), dtype=torch.float), dim=1)
    FP = torch.sum(torch.as_tensor(torch.logical_and(false, positive), dtype=torch.float), dim=1)
    FN = torch.sum(torch.as_tensor(torch.logical_and(true, negative), dtype=torch.float), dim=1)
    try:
        fp_rate = FP / (TN + FP)
        fn_rate = FN / (FN + TP)
    except Exception as err:
        print("ber(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:\n")
        print("Error ", err)
        print(f"TN : {TN}, FP : {FP}, FN: {FN}, TP: {TP}")
        exit(0)

    fp_rate[fp_rate != fp_rate] = .0
    fn_rate[fn_rate != fn_rate] = .0

    return torch.mean(.5 * (fp_rate + fn_rate)).item()


def false_negative_rate(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold
    output_flat = output.view(batch_size, -1) >= threshold

    true = target_flat
    positive = output_flat
    negative = torch.logical_not(output_flat)
    TP = torch.sum(torch.as_tensor(torch.logical_and(true, positive), dtype=torch.float), dim=1)
    FN = torch.sum(torch.as_tensor(torch.logical_and(true, negative), dtype=torch.float), dim=1)

    try:
        fnr = FN / (FN + TP)
        fnr[fnr != fnr] = .0
    except Exception as err:
        print("false_negative_rate\n")
        print("Error ", err)
        print(f"TP : {TP}, FN: {FN}")
        exit(0)

    return torch.mean(fnr).item()


def false_positive_rate(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold
    output_flat = output.view(batch_size, -1) >= threshold

    false = torch.logical_not(target_flat)
    positive = output_flat
    negative = torch.logical_not(output_flat)
    TN = torch.sum(torch.as_tensor(torch.logical_and(false, negative), dtype=torch.float), dim=1)
    FP = torch.sum(torch.as_tensor(torch.logical_and(false, positive), dtype=torch.float), dim=1)
    try:
        fpr = FP / (TN + FP)
        fpr[fpr != fpr] = .0
    except Exception as err:
        print("false_positive_rate\n")
        print("Error ", err)
        print(f"TN : {TN}, FP: {FP}")
        exit(0)

    return torch.mean(fpr).item()


# (TP + TN) / (TP + TN + FP + FN)  : 전체 중에서 Positive 맞춘것과 Negative라고 맞춘것의 합
def accuracy(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold
    output_flat = output.view(batch_size, -1) >= threshold
    equal = torch.as_tensor(output_flat == target_flat, dtype=torch.float)

    return torch.mean(torch.mean(equal, dim=1)).item()


# TP / TP + FN
# 미완성
def calc_recall(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    print(f"Incomplete....")
    return
    batch_size = output.shape[0]
    target_flat = target.view(batch_size, -1) >= threshold
    output_flat = output.view(batch_size, -1) >= threshold
    true = target_flat
    positive = output_flat
    negative = torch.logical_not(output_flat)
    TP = torch.sum(torch.as_tensor(torch.logical_and(true, positive), dtype=torch.float), dim=1)
    FN = torch.sum(torch.as_tensor(torch.logical_and(true, negative), dtype=torch.float), dim=1)
    
#    print(f"TP : {torch.sum(TP)}, FN: {torch.sum(FN)}")
    try:
        recall = TP / (TP + FN)
        print(recall)
    except Exception as err:
        print("false_negative_rate\n")
        print("Error ", err)
        print(f"TP : {TP}, FN: {FN}")
        exit(0)

    return torch.mean(recall).item()

def calc_psnr(output: torch.Tensor, target: torch.Tensor, mask_not_use: torch.Tensor) -> float:
    # peak_signal_noise_ratio
    # image1,2 have range [0, 1]
    mse = np.mean((output.numpy() - target.numpy())**2)
    if mse == 0:  
        # divide 0 if same image 
        return 100      
    return 20 * math.log10( 1.0 / math.sqrt(mse))    




'''
    Cals PSNR within Whole Image, Shadow Area (True Area in Mask) 
    and NonShadow Area (False Area in Mask)
    Important Input Value Range : [0, 1]
'''
def calc_psnr_in_shadow(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    # image1,2 have range [0, 1]
    # return all_image, mask_true area, mask_false area

    mask_true   = mask > 0.5        # shadow area
    mask_false  = mask < 0.5        # non shadow area
    
    mse_output      = np.mean((output.numpy() - target.numpy()) ** 2)
    if mse_output == 0:
        psnr_output = 100
    else:
        psnr_output = 20.0 * math.log10( 1.0 / math.sqrt(mse_output))   

    # -----------------------------------------------------------------------
    # shadow area
    no_true     = torch.sum(mask_true == True)
    output_true = torch.as_tensor(output * mask_true, dtype = torch.float)
    target_true = torch.as_tensor(target * mask_true, dtype = torch.float)
    true_abs    = torch.abs(output_true - target_true)

    if torch.sum(true_abs) == 0 or no_true == 0:
        psnr_output_true = 100
    else:
        mse_output_true = torch.sum (true_abs * true_abs) / no_true
#        print(f"mse_output_true: {mse_output_true}")
        psnr_output_true = 20.0 * math.log10( 1.0 / math.sqrt(mse_output_true))   
#        print(f"psnr_output_true: {psnr_output_true}")

    # -----------------------------------------------------------------------
    # non_shadow area
    no_false     = torch.sum(mask_false == True)
    output_false = torch.as_tensor(output * mask_false, dtype = torch.float)
    target_false = torch.as_tensor(target * mask_false, dtype = torch.float)
    false_abs    = torch.abs(output_false - target_false)
    if torch.sum(false_abs) == 0 or no_false == 0:
        psnr_output_false = 100
    else:
        mse_output_false = torch.sum (false_abs * false_abs) / no_false
        psnr_output_false = 20.0 * math.log10( 1.0 / math.sqrt(mse_output_false))   

    #-------------------------
    return psnr_output_true

'''
    Cals PSNR within Whole Image, Shadow Area (True Area in Mask) 
    and NonShadow Area (False Area in Mask)
    Important Input Value Range : [0, 1]
'''
def calc_psnr_in_non_shadow(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    # image1,2 have range [0, 1]
    # return all_image, mask_true area, mask_false area

    mask_true   = mask > 0.5        # shadow area
    mask_false  = mask < 0.5        # non shadow area
    
    mse_output      = np.mean((output.numpy() - target.numpy()) ** 2)
    if mse_output == 0:
        psnr_output = 100
    else:
        psnr_output = 20.0 * math.log10( 1.0 / math.sqrt(mse_output))   

    # -----------------------------------------------------------------------
    # shadow area
    no_true     = torch.sum(mask_true == True)
    output_true = torch.as_tensor(output * mask_true, dtype = torch.float)
    target_true = torch.as_tensor(target * mask_true, dtype = torch.float)
    true_abs    = torch.abs(output_true - target_true)

    if torch.sum(true_abs) == 0 or no_true == 0:
        psnr_output_true = 100
    else:
        mse_output_true = torch.sum (true_abs * true_abs) / no_true
#        print(f"mse_output_true: {mse_output_true}")
        psnr_output_true = 20.0 * math.log10( 1.0 / math.sqrt(mse_output_true))   
#        print(f"psnr_output_true: {psnr_output_true}")

    # -----------------------------------------------------------------------
    # non_shadow area
    no_false     = torch.sum(mask_false == True)
    output_false = torch.as_tensor(output * mask_false, dtype = torch.float)
    target_false = torch.as_tensor(target * mask_false, dtype = torch.float)
    false_abs    = torch.abs(output_false - target_false)
    if torch.sum(false_abs) == 0 or no_false == 0:
        psnr_output_false = 100
    else:
        mse_output_false = torch.sum (false_abs * false_abs) / no_false
        psnr_output_false = 20.0 * math.log10( 1.0 / math.sqrt(mse_output_false))   
        
    #-------------------------
    return psnr_output_false


'''
    Cals PSNR within Whole Image, Shadow Area (True Area in Mask) 
    and NonShadow Area (False Area in Mask)
    Important Input Value Range : [0, 1]
'''
def calc_psnr_within_mask(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    # image1,2 have range [0, 1]
    # return all_image, mask_true area, mask_false area

    mask_true   = mask > 0.5        # shadow area
    mask_false  = mask < 0.5        # non shadow area
    
    mse_output      = np.mean((output.numpy() - target.numpy()) ** 2)
    if mse_output == 0:
        psnr_output = 100
    else:
        psnr_output = 20.0 * math.log10( 1.0 / math.sqrt(mse_output))   

    # -----------------------------------------------------------------------
    # shadow area
    no_true     = torch.sum(mask_true == True)
    output_true = torch.as_tensor(output * mask_true, dtype = torch.float)
    target_true = torch.as_tensor(target * mask_true, dtype = torch.float)
    true_abs    = torch.abs(output_true - target_true)

    if torch.sum(true_abs) == 0 or no_true == 0:
        psnr_output_true = 100
    else:
        mse_output_true = torch.sum (true_abs * true_abs) / no_true
#        print(f"mse_output_true: {mse_output_true}")
        psnr_output_true = 20.0 * math.log10( 1.0 / math.sqrt(mse_output_true))   
#        print(f"psnr_output_true: {psnr_output_true}")

    # -----------------------------------------------------------------------
    # non_shadow area
    no_false     = torch.sum(mask_false == True)
    output_false = torch.as_tensor(output * mask_false, dtype = torch.float)
    target_false = torch.as_tensor(target * mask_false, dtype = torch.float)
    false_abs    = torch.abs(output_false - target_false)
    if torch.sum(false_abs) == 0 or no_false == 0:
        psnr_output_false = 100
    else:
        mse_output_false = torch.sum (false_abs * false_abs) / no_false
        psnr_output_false = 20.0 * math.log10( 1.0 / math.sqrt(mse_output_false))   
        
    #-------------------------
    return psnr_output, psnr_output_true, psnr_output_false




def denormalize_image(img, mean, std):
    demean = tuple(-m / s for m, s in zip(mean, std))
    destd = tuple(1 / s for s in std)

    demean = torch.tensor(demean, dtype=torch.float, device=img.device).view(1, -1, 1, 1)
    destd = torch.tensor(destd, dtype=torch.float, device=img.device).view(1, -1, 1, 1)

    return img.sub_(demean).div_(destd)


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
    

#    print(f"to_lab_space 0: {output[:,0,:,:].min()} {output[:,0,:,:].max()}") 
#    print(f"to_lab_space 1: {output[:,1,:,:].min()} {output[:,1,:,:].max()}") 
#    print(f"to_lab_space 2: {output[:,2,:,:].min()} {output[:,2,:,:].max()}") 

    return output.to(device)


def create_split_field_and_metric_funcs_dict(splits: Iterable[str], field_metric_funcs_dict: Dict[
    str, Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]]) \
        -> Dict[str, Dict[str, Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]]]:
    return dict(
        (split, dict(
            (f'{split}_{field}', metric_funcs)
            for field, metric_funcs in field_metric_funcs_dict.items()
        )) for split in splits
    )


def create_split_and_criterion_lists_dict(splits: Iterable[str], criterion_names: Iterable[str]) -> Dict[
    str, List[str]]:
    return dict(
        (split, [f'{split}_{criterion_name}' for criterion_name in criterion_names]) for split in splits
    )


def get_avg_metrics_and_losses(metric_containers, criterion_container):
    avg_metrics_and_losses = {}
    avg_losses = criterion_container.calc_averages()
    for field, metric_container in metric_containers.items():
        avg_metrics_and_losses[field] = metric_container.calc_averages()

    for criterion_field, loss in avg_losses.items():
        field = criterion_field[criterion_field.rfind('/') + 1:]
        if field not in avg_metrics_and_losses:
            avg_metrics_and_losses[field] = dict()
        avg_metrics_and_losses[field][criterion_field] = loss

    return avg_metrics_and_losses


    
class CalcMetric:
    def __init__(self):
        pass

    def prepare(self):
        pass
    
    def call(self):
        pass
        
    def finish(self):
        pass

class MetricCalculator:
    def __init__(self, metric_funcs, transform=None):
        self.metric_funcs = dict()
        self.transform = transform if transform is not None else (lambda x: x)
        self.add_metric_funcs(metric_funcs)

    def add_metric_func(self, metric_name, metric_func):
        self.metric_funcs[metric_name] = metric_func

    def add_metric_funcs(self, metric_funcs):
        if not isinstance(metric_funcs, dict):
            metric_funcs = dict((func.__name__, func) for func in metric_funcs)

        for name, func in metric_funcs.items():
            self.add_metric_func(name, func)

    def calc_metrics(self, output, target):
        output = self.transform(output)
        target = self.transform(target)
        return dict((name, func(output, target)) for name, func in self.metric_funcs.items())

    def calc_metrics_option(self, output, target, option ):
        output = self.transform(output)
        target = self.transform(target)
        option = self.transform(option)        
        return dict((name, func(output, target, option)) for name, func in self.metric_funcs.items())

    def __str__(self):
        nl = '\n'
        return f'transform: {str(self.transform)}' \
            + f'\nfunctions: {nl.join([str(key) for key in self.metric_funcs])}\n'


class MetricContainer:
    def __init__(self, field_name=None):
        self.container = dict()
        self.field_name = field_name

    def initialize(self):
        self.container = dict()

    def append_metric(self, metric_key, metric_value):
        if metric_key in self.container:
            self.container[metric_key].append(metric_value)
        else:
            self.container[metric_key] = [metric_value]

    def append_metrics(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.append_metric(key, value)

    def calc_average(self, metric_key):
        return sum(self.container[metric_key]) / float(len(self.container[metric_key]))

    def calc_averages(self):
        if self.field_name is None:
            return dict((key, self.calc_average(key)) for key in self.container)
        else:
            return dict((f'{self.field_name}/{key}', self.calc_average(key)) for key in self.container)

    def __str__(self):
        return f'field name: {self.field_name}\n' + \
            '\n'.join([str(key) + ": " + str(values) for key, values in self.container]) + '\n'

class Test_MetricWriterLoop: #(BaseLoop): #, option.OptionParser):
    @staticmethod
    def add_options(parser):
        parser.add_argument('--print_frequency', type=int, default=25)
        parser.add_argument('--result_dir', type=str, default='result_dir/')
        return parser

    def __init__(self, metric_names_with_field, criterion_names, loop_name='epoch', iter_name='batch', 
		lab_name='', instance_name='', comment='', prefix='', print_on_tensorboard=True, tensorboard_writer=None, no_parent=False):
        super(Test_MetricWriterLoop, self).__init__()
        self.metric_names_with_field = metric_names_with_field
        self.criterion_names = criterion_names

        self.metric_containers: Dict[str, Dict[str, List[float]]] = None
        self.loss_containers: Dict[str, List[float]] = None
        self._initialize_containers()

        self.prefix = prefix
        self.loop_name = loop_name
        self.iter_name = iter_name
        self.lab_name = lab_name
        self.instance_name = instance_name 

    def _initialize_containers(self):
        self.metric_containers = dict()
        for field, metric_names in self.metric_names_with_field.items():
            self.metric_containers[field] = dict()
            for metric_name in metric_names:
                self.metric_containers[field][metric_name] = list()
        self.loss_containers = dict((criterion, list()) for criterion in self.criterion_names)
#		print(f"metric_containers: {self.metric_containers}")) 
#		print(f"loss_containers: {self.loss_containers}")

    def after_iteration(self, calculated_metrics, calculated_losses):
        # insert losses
        self._insert_losses(calculated_losses)
        self._insert_metrics(calculated_metrics)

        #if (self.index + 1) % self.print_frequency == 0:
        #j    # calculate metrics
        #    metrics_avg = self.calc_metrics_avg()
        #    losses_avg = self.calc_losses_avg()
        #    merged_avg = self._merge_avg_metrics_and_losses(metrics_avg, losses_avg)
        #    self._print_and_write_iter_log(merged_avg, self.index + 1, self.get_parent_index())

    def calc_metrics_avg(self) -> Dict[str, Dict[str, float]]:
        avgs = dict()
        for field, func_names in self.metric_containers.items():
            avgs[field] = dict()
            for func_name, metrics in func_names.items():
                metrics = self.metric_containers[field][func_name]
                avgs[field][func_name] = sum(metrics) / len(metrics)
        return avgs

    def calc_losses_avg(self) -> Dict[str, float]:
        avgs = dict()
        for criterion, losses in self.loss_containers.items():
            avgs[criterion] = sum(losses) / len(losses)
        return avgs

    def _insert_metrics(self, metrics_with_name_with_field):
        for field, metrics_with_name in metrics_with_name_with_field.items():
            for name, metric in metrics_with_name.items():
                self.metric_containers[field][name].append(metric)

    def _insert_losses(self, losses_with_field):
        for field, loss in losses_with_field.items():
            self.loss_containers[field].append(loss)

    def _merge_avg_metrics_and_losses(self, metrics_avg, losses_avg):
        metric_dict = dict()
        prefix = self.prefix + "_" if self.prefix != "" else ""
        for field, metrics_with_name in metrics_avg.items():
            for metric_name, avg_metric in metrics_with_name.items():
                metric_dict[f'{prefix}{field}/{metric_name}'] = avg_metric
        for loss_name, avg_losses in losses_avg.items():
            metric_dict[f'{prefix}criterion/{loss_name}'] = avg_losses
        return metric_dict

    def print_data(self):
        print(f"print_data")
        print(self.metric_containers)
        print(self.loss_containers)


if __name__ == '__main__':
    #torch.set_printoptions(precision=50)    
    input = torch.sigmoid(torch.randn(1, 2, 2))
    #target = torch.sigmoid(input+torch.randn(1, 2, 2)/10.0)
    print(">>> input : ", input.view(-1))
    target = input
    print(">>> target: ", target.view(-1))
    target[0,0,0] = input[0,0,0] + 0.01   
    print("==============")
    print(">>> input : ", input.view(-1))
    print(">>> target: ", target.view(-1)) 

    input[0,0,1] = 0.01   
    
    
    mask = torch.randint(2, size=(1, 2, 2))
    mask = mask == 1

    print(">>> input : ", input.view(-1))
    print(">>> target: ", target.view(-1))
    print(mask.view(-1))


    psnr_all, psnr_shadow, psnr_non_shadow = calc_psnr_within_mask(input, target, mask)
    
    print("PSNR: ", psnr_all, psnr_shadow, psnr_non_shadow)

    
    input = torch.as_tensor(input * mask, dtype=torch.float)
    #target = torch.as_tensor(target * mask, dtype=torch.float)
    target = target * mask
    no = torch.sum(mask == True)
    diff = torch.abs(input-target) 
    diff_square = torch.sum(torch.abs(input-target) * torch.abs(input-target))
    print(f"diff--{diff}")
    print(f"diff_square--{diff_square}\n>> mean : {diff_square/no} --torch.mean {torch.mean(diff*diff)}")
    
    
    



    m_arr = zip(input, target, mask)
    print(m_arr)
    for i in m_arr:
        print(i)

    #loss = mse_loss(input, target)
    #print(f"--> {loss:.3f}")
    #list_input = list(input.view(-1).numpy())
    


