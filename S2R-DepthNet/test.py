# https://docs.opensource.microsoft.com/content/releasing/copyright-headers.html
import os 
import torch
import loaddata
import matplotlib
import numpy as np
from utils import *
import matplotlib.cm
import torch.nn as nn
import DSAModules

import torch.nn.parallel
import matplotlib as mpl
from models import modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torch.autograd import Variable
import torch.backends.cudnn as cudnn

plt.switch_backend('agg')
plt.set_cmap("jet")

# =========================== Parameters =================
# 在这里定义参数，方便在 VSCode 中直接运行
params = {
    'dataset': 'VKITTI',                         # 数据集名称
    'root': '/path/to/dataset',                  # 数据集路径
    'test_datafile': '/path/to/test_data.txt',   # 测试数据文件路径
    'batchSize': 1,                              # 批次大小
    'nThreads': 8,                               # 数据加载线程数
    'loadSize': 286,                             # 图像加载尺寸
    'out_dir': 'out',                            # 输出目录
    'Shared_Struct_Encoder_path': '/path/to/Shared_Struct_Encoder.pth',
    'Struct_Decoder_path': '/path/to/Struct_Decoder.pth',
    'DepthNet_path': '/path/to/DepthNet.pth',
    'DSAModle_path': '/path/to/DSAModle.pth'
}

def save_test(handle, result1_log):
    # 省略该函数的实现...

def kitti_metrics_preprocess(pred, gt):
    # 省略该函数的实现...

def kitti_compute_metrics(pred, gt):
    # 省略该函数的实现...

def nyu_compute_metrics(pred, gt):
    # 省略该函数的实现...

def nyu_metrics_preprocess(pred, gt):
    # 省略该函数的实现...

def main(params):
    print("Loading the dataset ...")

    real_loader = loaddata.create_test_dataloader(
        dataset=params['dataset'],
        root=params['root'],
        data_file=params['test_datafile'],
        batchsize=params['batchSize'],
        nThreads=params['nThreads'],
        loadSize=params['loadSize']
    )
    
    print("Loading data set is complete!")
    print("=======================================================================================")
    print("Building models ...")

    # Define Shared Structure Encoder
    Shared_Struct_Encoder = modules.Struct_Encoder(n_downsample=2, n_res=4, 
                                                input_dim=3, dim=64, 
                                                norm='in', activ='lrelu', 
                                                pad_type='reflect')
    
    # Define Structure Decoder
    Struct_Decoder = modules.Struct_Decoder()

    # Define Depth-specific Attention (DSA) module
    Attention_Model = DSAModules.drn_d_22(pretrained=True)
    DSAModle = DSAModules.AutoED(Attention_Model)

    # Define DepthNet
    DepthNet = modules.Depth_Net()
    init_weights(DepthNet, init_type='normal')
    
    # Move models to GPU
    Shared_Struct_Encoder = Shared_Struct_Encoder.cuda()    
    Struct_Decoder = torch.nn.DataParallel(Struct_Decoder).cuda()
    DSAModle = torch.nn.DataParallel(DSAModle).cuda()
    DepthNet = torch.nn.DataParallel(DepthNet).cuda()    
    
    # Load models
    Shared_Struct_Encoder.load_state_dict(torch.load(params['Shared_Struct_Encoder_path']))
    Struct_Decoder.load_state_dict(torch.load(params['Struct_Decoder_path']))
    DSAModle.load_state_dict(torch.load(params['DSAModle_path']))
    DepthNet.load_state_dict(torch.load(params['DepthNet_path']))
    
    if not os.path.exists(params['out_dir']):
        os.mkdir(params['out_dir'])
    
    if params['dataset'] == "KITTI":
        Shared_Struct_Encoder.eval()
        Struct_Decoder.eval()
        DSAModle.eval()
        DepthNet.eval()

        result_log = [[] for _ in range(7)]

        for step, real_batched in enumerate(real_loader, start=1):
            print("step:", step)
            image, depth_, depth_interp_ = real_batched['left_img'], real_batched['depth'], real_batched['depth_interp']

            image = torch.autograd.Variable(image).cuda()
            depth_ = torch.autograd.Variable(depth_).cuda()

            # predict
            struct_code = Shared_Struct_Encoder(image)
            structure_map = Struct_Decoder(struct_code)
            attention_map = DSAModle(image)
            depth_specific_structure = attention_map * structure_map
            pred_depth = DepthNet(depth_specific_structure)
            pred_depth = torch.nn.functional.interpolate(pred_depth[-1], size=[depth_.size(1),depth_.size(2)], mode='bilinear',align_corners=True)
    
            pred_depth_np = np.squeeze(pred_depth.cpu().detach().numpy())
            gt_np = np.squeeze(depth_.cpu().detach().numpy())
            depth_interp_np = np.squeeze(depth_interp_.cpu().detach().numpy())

            pred_depth_np += 1.0
            pred_depth_np /= 2.0
            pred_depth_np *= 80.0

            test_result = kitti_compute_metrics(pred_depth_np, gt_np)  # list1 

            for it, item in enumerate(test_result):
                result_log[it].append(item)

        with open(params['out_dir'] + "/evalog.txt", 'w') as f:
            f.write('Done testing -- epoch limit reached\n')
            f.write(f"after {step} iterations\n\n")
            save_test(f, result_log)

    elif params['dataset'] == "NYUD_V2":
        # 省略相同处理逻辑...
        pass

if __name__ == '__main__':
    main(params)
