import os
import torch
import numpy as np
import modules
import DSAModules
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize,Resize
from utils import *

# =========================== Parameters =================
params = {
    'root': '/autodl-fs/data/data/data/MORE/img_org/total',         # 图像数据路径
    'batchSize': 16,                               # 批次大小
    'nThreads': 8,                                # 数据加载线程数
    'loadSize': (512, 512),                              # 图像加载尺寸
    'out_dir': '/autodl-fs/data/dep',                             # 输出目录
    'Shared_Struct_Encoder_path': './S2R-DepthNet\checkpoints\struct_encoder_suncg.pth',
    'Struct_Decoder_path': './S2R-DepthNet\checkpoints\struct_decoder_suncg.pth',
    'DepthNet_path': './S2R-DepthNet\checkpoints\depthnet_suncg.pth',
    'DSAModle_path': './S2R-DepthNet\checkpoints\dsamodels_suncg.pth'
}

img_transform = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])
class MyDataset(Dataset):
    def __init__(self, root, img_transform=None):
        """
        root: 图像文件夹的根目录
        img_transform: 图像变换（例如ToTensor，Normalize等）
        """
        self.root = root
        self.img_transform = img_transform

        # 获取图像文件列表，支持常见图像格式
        self.files = [file for file in os.listdir(self.root) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 获取图像路径
        img_path = os.path.join(self.root, self.files[index])
        assert os.path.exists(img_path), f"Image does not exist at path: {img_path}"
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')  # 确保为RGB格式

        # 应用图像变换
        if self.img_transform is not None:
            image = self.img_transform(image)

        return image, self.files[index]  # 返回图像和图像文件名
    

# 加载图像数据
def load_images(image_folder):
    images = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, file_name)
            img = plt.imread(img_path)  # 使用matplotlib加载图像
            images.append((file_name, img))
    return images

def main(params):
    print("Loading images ...")
    # images = load_images(params['root'])
    dataset = MyDataset(root=params['root'], img_transform=img_transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 加载预训练模型
    print("Loading models ...")
    Shared_Struct_Encoder = modules.Struct_Encoder(n_downsample=2, n_res=4, 
                                                   input_dim=3, dim=64, 
                                                   norm='in', activ='lrelu', 
                                                   pad_type='reflect').cuda()
    Struct_Decoder = modules.Struct_Decoder()
    Struct_Decoder = torch.nn.DataParallel(Struct_Decoder).cuda()
    Attention_Model = DSAModules.drn_d_22(pretrained=True)
    DSAModle = torch.nn.DataParallel(DSAModules.AutoED(Attention_Model)).cuda()
    DepthNet = modules.Depth_Net()
    init_weights(DepthNet, init_type='normal')

    DepthNet = torch.nn.DataParallel(DepthNet).cuda()

    Shared_Struct_Encoder.load_state_dict(torch.load(params['Shared_Struct_Encoder_path']))
    Struct_Decoder.load_state_dict(torch.load(params['Struct_Decoder_path']))
    DSAModle.load_state_dict(torch.load(params['DSAModle_path']))
    DepthNet.load_state_dict(torch.load(params['DepthNet_path']))

    # 评估模式
    Shared_Struct_Encoder.eval()
    Struct_Decoder.eval()
    DSAModle.eval()
    DepthNet.eval()

    # 推理过程
    os.makedirs(params['out_dir'], exist_ok=True)
    for images, file_names in data_loader:
        images = images.cuda()
        
        print("Batch file names:", file_names)
        print("Batch image tensor size:", images.size())
    # # print(f"Loaded {len(images)} images.")
    #     img_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0  # 转为张量并归一化
    #     # img_tensor = torch.tensor(img).float().unsqueeze(0).cuda() / 255.0  # 转为张量并归一化

        with torch.no_grad():
            # 前向推理
            struct_code = Shared_Struct_Encoder(images) # torch.Size([1, 256, 100, 150])
            structure_map = Struct_Decoder(struct_code) # torch.Size([1, 1, 400, 600])
            attention_map = DSAModle(images) # torch.Size([1, 1, 400, 600])
            depth_specific_structure = attention_map * structure_map # [1,1,400,600]
            pred_depth = DepthNet(depth_specific_structure)
            pred_depth = torch.nn.functional.interpolate(pred_depth[-1], size=[608,608], mode='bilinear',align_corners=True)

            pred_depth_np = np.squeeze(pred_depth.cpu().numpy())
            for i, file_name in enumerate(file_names):
                image_id = os.path.splitext(file_name)[0]  # 获取图像ID
                output_path = os.path.join(params['out_dir'], f"{image_id}.npz")
                
                # 保存到 npz 文件中
                np.savez_compressed(output_path, depth_map=pred_depth_np[i])
                print(f"Saved depth map to {output_path}")

            # 可视化
            concatenated_image = np.concatenate(pred_depth_np, axis=1)  # 拼接批次中的深度图
            plt.imshow(concatenated_image, cmap='jet')
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    main(params)