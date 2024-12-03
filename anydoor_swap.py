import cv2
import os
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention


from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import tqdm
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='swap.log',  # 指定日志文件名
                    filemode='w')  # 'w' 为覆盖模式，'a' 为追加模式


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

device = "cuda:0"

config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# model = None
# ddim_sampler = None

# 加载 YOLOv5 模型
model_yolo = YOLO('/root/autodl-tmp/yolov5lu.pt')
sam_checkpoint = "/root/autodl-tmp/sam_vit_h_4b8939.pth" # 替换为您的权重文件路径
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

label_mapping = model_yolo.names

def set_args(input_img = None,input_folder = None, output_folder = None):
    class Args:
        def __init__(self):
            self.input_img = input_img
            self.input_folder = input_folder
            self.output_folder = output_folder
            self.device = device
            self.from_file = False
            self.from_img = False
    return Args()

def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask
def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
    return item

def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image

def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale = 5.0):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)

    # ====
    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 100 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    return gen_image

def normalize(value, min_value, max_value):
    """
    对单个指标进行归一化。
    """
    if max_value == min_value:  # 防止除零
        return 0.0
    return (value - min_value) / (max_value - min_value)
def extract_objects_with_irregular_mask(image, masks):
    """
    使用不规则掩码从图像中提取目标。

    Args:
        image (numpy.ndarray): 原始图像 (H, W, 3)。
        masks (list): SAM 模型预测的掩码列表，每个掩码为 (H, W)。

    Returns:
        list: 每个元素为裁剪后的目标 (裁剪图像, 掩码)。
    """
    obj_images = []
    intermediate_dir = "./tmp_mask"

    for i, mask in enumerate(masks):
        # 创建一个空白 RGBA 图像
        rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

        # 应用掩码提取目标
        rgba_image[:, :, :3] = image  # 复制 RGB 通道
        rgba_image[:, :, 3] = (mask * 255).astype(np.uint8)  # 将掩码作为 Alpha 通道


        # 裁剪目标区域
        # cropped_image = rgba_image[y1:y2, x1:x2]
        save_mask_path = os.path.join(intermediate_dir, f"mask_{i}.png")
        save_obj_path = os.path.join(intermediate_dir, f"obj_{i}.png")
        # 保存裁剪后的结果
        rgb_image = rgba_image[:,:,:3]
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        mask = (rgba_image[:,:,3]>128).astype(np.uint8)
        obj_images.append((rgb_image, mask))
        cv2.imwrite(save_obj_path, rgba_image)
        cv2.imwrite(save_mask_path, mask)

    return obj_images
def extract_objects_with_sam(image, bbox, predictor):
    """
    利用 YOLOv5 提取的边界框和 SAM 模型生成目标掩码。

    Args:
        image (numpy.ndarray): 原始图像 (H, W, 3)。
        bbox (tuple): 边界框 (x1, y1, x2, y2)。
        predictor: SAM 模型的预测器对象。

    Returns:
        numpy.ndarray: 仅保留目标的裁剪图像，其余区域全黑。
        numpy.ndarray: SAM 生成的目标掩码。
    """
    x1, y1, x2, y2 = bbox

    cropped_image = image[y1:y2, x1:x2]
    # 使用 SAM 生成目标的掩码
    predictor.set_image(cropped_image)
    input_points = np.array([[(x2 - x1) // 2, (y2 - y1) // 2]])  # 中心点
    input_labels = np.array([1])
    masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)

    # 提取目标掩码（取第一个）
    mask = masks[2]
    cropped_image[mask == 0] = 0
    # 裁剪目标区域
    # cropped_image = image[y1:y2, x1:x2]
    # 构造全黑背景的目标图像
    cropped_mask = (mask > 128).astype(np.uint8)  # 转为二值化掩码

    cv2.imwrite("cropped_result.png", cropped_image)
    cv2.imwrite("cropped_mask.png", cropped_mask)

    return cropped_image, cropped_mask
# def extract_and_swap_objects(image_path,save_path):
#     """
#     提取具有相同标签的两个目标及其掩码，并交换位置。

#     Args:
#         image_path (str): 输入图像路径。
#         save_path (str): 输出图像保存路径。

#     Returns:
#         None: 如果没有符合条件的目标，直接跳过。
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Image not found:", image_path)
#         return

#     # 推理检测
#     results = model_yolo(image)

#     # 加载 SAM 模型
#     predictor.set_image(image)
#     # 获取类别映射
#     label_mapping = model_yolo.names  # {0: 'person', 1: 'bicycle', 2: 'car', ...}
#     # 提取检测结果，按标签分组
#     objects = {}
#     for r in results:
#         for box in r.boxes:
#             label = int(box.cls[0])  # 类别
#             confidence = box.conf[0]  # 置信度
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中心点
#             area = (x2 - x1) * (y2 - y1)  # 计算边界框面积
#             aspect_ratio = (x2 - x1) / (y2 - y1)  # 计算宽高比

#             # 保存到按类别分组的字典
#             if label not in objects:
#                 objects[label] = []
#             objects[label].append({
#                 "area": area,
#                 "aspect_ratio": aspect_ratio,
#                 "bbox": (x1, y1, x2, y2),
#                 "confidence": confidence,
#                 "center": (cx, cy)
#             })

#     # 检查是否存在至少两个目标具有相同标签
#     tag = True
#     label_name = None
#     image_center = (image.shape[1] // 2, image.shape[0] // 2)  # 图像中心点
#     # 综合排序权重
#     AREA_WEIGHT = 1.0
#     SHAPE_WEIGHT = 1.0
#     CENTER_WEIGHT = 1.0
#     # 提取指标范围用于归一化
#     all_areas = [obj["area"] for objs in objects.values() for obj in objs]
#     all_ratios = [obj["aspect_ratio"] for objs in objects.values() for obj in objs]
#     all_centers = [
#         np.linalg.norm(np.array(obj["center"]) - np.array(image_center))
#         for objs in objects.values() for obj in objs
#     ]

#     min_area, max_area = min(all_areas), max(all_areas)
#     min_ratio, max_ratio = min(all_ratios), max(all_ratios)
#     min_center, max_center = min(all_centers), max(all_centers)
#     for label, objs in objects.items():
#         if len(objs) < 2:
#             tag = False
#             label_name = None
#             continue  # 如果目标数少于 2，跳过此标签
#         else:
#             tag = True
#         if tag == True:
#             # 对所有目标进行两两组合打分
#             min_score = float('inf')
#             obj1, obj2 = None, None
#             for i in range(len(objs)):
#                 for j in range(i + 1, len(objs)):
#                     o1, o2 = objs[i], objs[j]
#                         # 归一化后计算指标
#                     area_diff = normalize(abs(o1["area"] - o2["area"]), min_area, max_area)
#                     shape_diff = normalize(abs(o1["aspect_ratio"] - o2["aspect_ratio"]), min_ratio, max_ratio)
#                     center1 = np.linalg.norm(np.array(o1["center"]) - np.array(image_center))
#                     center2 = np.linalg.norm(np.array(o2["center"]) - np.array(image_center))
#                     center_score = normalize(center1 + center2, min_center, max_center)

#                     # 综合得分
#                     score = (
#                         AREA_WEIGHT * area_diff +
#                         SHAPE_WEIGHT * shape_diff +
#                         CENTER_WEIGHT * center_score
#                     )

#                     if score < min_score:
#                         min_score = score
#                         obj1, obj2 = o1, o2
#             # 使用 SAM 提取目标和掩码
#             # cropped1, mask1 = extract_objects_with_sam(image, obj1["bbox"], predictor)
#             # cropped2, mask2 = extract_objects_with_sam(image, obj2["bbox"], predictor)

#             # 使用 SAM 生成背景掩码
#             bg_masks = []
#             intermediate_dir = "./tmp_mask"
#             for i,obj in enumerate([obj1, obj2]):
#                 input_points = np.array([obj["center"]])
#                 input_labels = np.array([1])
#                 bg_mask, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)
#                 bg_masks.append(bg_mask[2])
#                     # 使用 YOLOv5 的边界框生成掩码
#             # masks = []
#             # intermediate_dir = "./tmp_mask"
#             # for i, obj in enumerate([obj1, obj2]):
#             #     x1, y1, x2, y2 = obj["bbox"]
#             #     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#             #     mask[y1:y2, x1:x2] = 1  # 将边界框区域设为 1
#             #     masks.append(mask)

#                 # 保存每个目标的掩码
#                 mask_path = os.path.join(intermediate_dir, f"bg_mask_{i}.png")
#                 cv2.imwrite(mask_path, bg_masks[i].astype(np.uint8) * 255)
#                 print(f"Saved mask {i} to {mask_path}")
#             # 裁剪目标区域
#             obj_images = extract_objects_with_irregular_mask(image, bg_masks)

#             # 获取背景图像（去除目标区域）
#             background = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

#             # 调用 AnyDoor 交换位置
#             gen_image1 = inference_single_image(cropped1, mask1, background.copy(), bg_masks[1])
#             background = gen_image1.astype(np.uint8)
#             gen_image2 = inference_single_image(cropped2, mask2, background.copy(), bg_masks[0])

#             # 保存结果
#             result_image = cv2.cvtColor(gen_image2, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(save_path, result_image)
#             print(f"Saved swapped image to {save_path}")
#             break
#     return  tag,label_name
def extract_and_swap_objects(image_path,save_path):
    """
    提取具有相同标签的两个目标及其掩码，并交换位置。

    Args:
        image_path (str): 输入图像路径。
        save_path (str): 输出图像保存路径。

    Returns:
        None: 如果没有符合条件的目标，直接跳过。
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found:", image_path)
        return

    # 推理检测
    results = model_yolo(image)

    # 加载 SAM 模型
    predictor.set_image(image)

    # 提取检测结果，按标签分组
    objects = {}
    for r in results:
        for box in r.boxes:
            label = int(box.cls[0])  # 类别
            label_name = label_mapping[label]
            confidence = box.conf[0]  # 置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边界框坐标
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 中心点

            # 保存到按类别分组的字典
            if label not in objects:
                objects[label] = []
            objects[label].append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "center": (cx, cy)
            })

    # 检查是否存在至少两个目标具有相同标签
    for label, objs in objects.items():
        if len(objs) < 2 or label_name == "person":
            continue  # 如果目标数少于 2，跳过此标签

        # 选择两个目标
        obj1, obj2 = objs[:2]

        # 使用 SAM 生成掩码
        masks = []
        for obj in [obj1, obj2]:
            input_points = np.array([obj["center"]])
            input_labels = np.array([1])
            mask, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)

            masks.append(mask[2])
        # 使用SAM 进行目标分割
        # 裁剪目标区域
        obj_images = []
        for i, obj in enumerate([obj1, obj2]):
            x1, y1, x2, y2 = obj["bbox"]
            cropped_image = image[y1:y2, x1:x2]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_mask = masks[i][y1:y2, x1:x2]
            cropped_mask = cropped_mask.astype(np.uint8)
            obj_images.append((cropped_image, cropped_mask, obj["bbox"]))

        # 获取背景图像（去除目标区域）
        bg_mask = 1 - (masks[0] + masks[1])
        background = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # 调用 AnyDoor 交换位置
        gen_image1 = inference_single_image(obj_images[0][0], obj_images[0][1], background.copy(), masks[1])
        # background = cv2.cvtColor(gen_image1.astype(np.uint8), cv2.COLOR_RGB2BGR)
        background = gen_image1.astype(np.uint8)
        gen_image2 = inference_single_image(obj_images[1][0], obj_images[1][1], background.copy(), masks[0])

        # 保存结果
        result_image = cv2.cvtColor(gen_image2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_image)
        print(f"Saved swapped image to {save_path}")
        return  # 只处理第一组，退出函数

    print("No matching objects found with at least two instances having the same label.")
def batch_process_images(input_dir, output_dir):
    """
    批量处理目录中的图像，提取两个具有相同类别的目标并交换位置。

    Args:
        input_dir (str): 输入图像目录路径。
        output_dir (str): 输出图像目录路径。
        sam_checkpoint (str): SAM 模型权重路径。

    Returns:
        None
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取目录中的所有图像文件
    image_files = [
        os.path.join(input_dir, file) for file in os.listdir(input_dir)
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    processed_img_count = 0
    # 遍历所有图像文件并处理
    for image_path in tqdm(image_files, desc="Processing Images"):
        try:
            # 构造输出路径
            image_name = os.path.basename(image_path)
            save_path = os.path.join(output_dir, f"{image_name}")

            # 调用单张图像的处理函数
            tag,label = extract_and_swap_objects(image_path, save_path)
            if tag == True:
                processed_img_count += 1
                logging.info(f"Processed {image_name}")
                logging.info(f"Processes Objects: {label}")
                logging.info(f"saved to {save_path}")
                logging.info(f"Processed {processed_img_count} images.")

            else:
                logging.warning(f"Skipping {image_path} due to insufficient objects.")
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")

if __name__ == '__main__': 
    
    # ==== Example for inferring a single image ===
    save_path = './output'
    image_path =r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\img_org\total\0ce97a65-feb3-52ce-b4e1-dac18cb90a9f.jpg"
    extract_and_swap_objects(image_path,save_path)


    # reference_image_path = './examples/TestDreamBooth/FG/01.png'
    # bg_image_path = './examples/TestDreamBooth/BG/000000309203_GT.png'
    # bg_mask_path = './examples/TestDreamBooth/BG/000000309203_mask.png'
    # save_path = './examples/TestDreamBooth/GEN/gen_res.png'

    # # reference image + reference mask
    # # You could use the demo of SAM to extract RGB-A image with masks
    # # https://segment-anything.com/demo
    # image = cv2.imread( reference_image_path, cv2.IMREAD_UNCHANGED)
    # mask = (image[:,:,-1] > 128).astype(np.uint8)
    # # 可视化mask，展示图片
    # cv2.imshow('mask', mask)
    # image = image[:,:,:-1]
    # image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    # ref_image = image 
    # ref_mask = mask

    # # background image
    # back_image = cv2.imread(bg_image_path).astype(np.uint8)
    # back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

    # # background mask 
    # tar_mask = cv2.imread(bg_mask_path)[:,:,0] > 128
    # tar_mask = tar_mask.astype(np.uint8)
    
    # gen_image = inference_single_image(ref_image, ref_mask, back_image.copy(), tar_mask)
    # h,w = back_image.shape[0], back_image.shape[0]
    # ref_image = cv2.resize(ref_image, (w,h))
    # vis_image = cv2.hconcat([ref_image, back_image, gen_image])
    
    # cv2.imwrite(save_path, vis_image [:,:,::-1])
    
#     #'''
#     # ==== Example for inferring VITON-HD Test dataset ===

#     from omegaconf import OmegaConf
#     import os 
#     DConf = OmegaConf.load('./configs/datasets.yaml')
#     save_dir = './VITONGEN'
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)

#     test_dir = DConf.Test.VitonHDTest.image_dir
#     image_names = os.listdir(test_dir)
    
#     for image_name in image_names:
#         ref_image_path = os.path.join(test_dir, image_name)
#         tar_image_path = ref_image_path.replace('/cloth/', '/image/')
#         ref_mask_path = ref_image_path.replace('/cloth/','/cloth-mask/')
#         tar_mask_path = ref_image_path.replace('/cloth/', '/image-parse-v3/').replace('.jpg','.png')

#         ref_image = cv2.imread(ref_image_path)
#         ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

#         gt_image = cv2.imread(tar_image_path)
#         gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

#         ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

#         tar_mask = Image.open(tar_mask_path ).convert('P')
#         tar_mask= np.array(tar_mask)
#         tar_mask = tar_mask == 5

#         gen_image = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask)
#         gen_path = os.path.join(save_dir, image_name)

#         vis_image = cv2.hconcat([ref_image, gt_image, gen_image])
#         cv2.imwrite(gen_path, vis_image[:,:,::-1])
#     #'''

    

