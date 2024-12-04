import torch
import sys
import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utilss import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point


import cv2
from PIL import Image
from ultralytics import YOLO
import logging

model_yolo = YOLO('/root/autodl-tmp/yolov5lu.pt')
sam_checkpoint = "/root/autodl-tmp/sam_vit_h_4b8939.pth" 
lama_ckpt = "/root/autodl-tmp/big-lama"

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./swap_log.log',
                    filemode='w')

def build_r_s_model():
    model_yolo = YOLO('/root/autodl-tmp/yolov5lu.pt')
    return model_yolo
def setup_args():
    class Args:
        def __init__(self):
            # self.input_img = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\img_org\total\000a91a4-7612-5842-9704-55c95894ce92.jpg"  # replace with your input image
            # self.input_img = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\img_org\total\0ce97a65-feb3-52ce-b4e1-dac18cb90a9f.jpg"
            # self.input_img = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\img_org\total\1f993990-0666-5f54-9de8-957abfcb93d7.jpg"
            self.input_folder = "/autodl-fs/data/data/data/MORE/img_org/total"
            self.output_dir = "/root/autodl-fs/swap_res"
            self.coords_type = "key_in"
            self.point_labels = [1]
            self.dilate_kernel_size = 15
            self.sam_model_type = "vit_h"
            self.sam_ckpt = sam_checkpoint  
            self.lama_config = "./lama/configs/prediction/default.yaml"
            self.lama_ckpt = lama_ckpt  

    return Args()
def normalize(value, min_value, max_value):
    """
    对单个指标进行归一化。
    """
    if max_value == min_value:  
        return 0.0
    return (value - min_value) / (max_value - min_value)
def is_overlapping(bbox1, bbox2):
    """ 检查两个矩形框是否重叠 """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

def get_clicked_points(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found:", image_path)
        return None, None, None, None, None, None


    results = model_yolo(image)


    label_mapping = model_yolo.names  # {0: 'person', 1: 'bicycle', 2: 'car', ...}

    objects = {}
    for r in results:
        for box in r.boxes:
            label = int(box.cls[0])  
            confidence = box.conf[0]  
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  
            area = (x2 - x1) * (y2 - y1)  
            aspect_ratio = (x2 - x1) / (y2 - y1)  # 宽高比

            if label not in objects:
                objects[label] = []
            objects[label].append({
                "area": area,
                "aspect_ratio": aspect_ratio,
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "center": (cx, cy)
            })


    tag = True
    label_name = None
    image_center = (image.shape[1] // 2, image.shape[0] // 2)  

    AREA_WEIGHT = 1.0
    SHAPE_WEIGHT = 1.0
    CENTER_WEIGHT = 1.0

    all_areas = [obj["area"] for objs in objects.values() for obj in objs]
    all_ratios = [obj["aspect_ratio"] for objs in objects.values() for obj in objs]
    all_centers = [
        np.linalg.norm(np.array(obj["center"]) - np.array(image_center))
        for objs in objects.values() for obj in objs
    ]

    min_area, max_area = min(all_areas), max(all_areas)
    min_ratio, max_ratio = min(all_ratios), max(all_ratios)
    min_center, max_center = min(all_centers), max(all_centers)
    for label, objs in objects.items():
        label_name = label_mapping[label]
        if len(objs) < 2 or label_name != 'person':
            tag = False
            label_name = None
            continue  
        else:
            tag = True
        if tag == True:

            logging.info(f"swapping objects label: {label_name}")
            min_score = float('inf')
            obj1, obj2 = None, None
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    o1, o2 = objs[i], objs[j]

                    area_diff = normalize(abs(o1["area"] - o2["area"]), min_area, max_area)
                    shape_diff = normalize(abs(o1["aspect_ratio"] - o2["aspect_ratio"]), min_ratio, max_ratio)
                    center1 = np.linalg.norm(np.array(o1["center"]) - np.array(image_center))
                    center2 = np.linalg.norm(np.array(o2["center"]) - np.array(image_center))
                    center_score = normalize(center1 + center2, min_center, max_center)

                    # 综合得分
                    score = (
                        AREA_WEIGHT * area_diff +
                        SHAPE_WEIGHT * shape_diff +
                        CENTER_WEIGHT * center_score
                    )

                    if score < min_score:
                        min_score = score
                        obj1, obj2 = o1, o2    
            #左上角坐标
            obj1_left_top = (obj1['bbox'][0], obj1['bbox'][1])
            obj2_left_top = (obj2['bbox'][0], obj2['bbox'][1])
            return obj1['center'], obj2['center'], obj1['bbox'], obj2['bbox'],obj1_left_top,obj2_left_top
    return None, None, None, None, None, None
def remove_single_obj(img, coords, bbox, args):
    latest_coords = coords
    if type(img) != np.ndarray:
        img = load_img_to_array(args.input_img)
    obj = img
    # obj = cv2.cvtColor(obj, cv2.COLOR_RGB2BGR)
    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        args.point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)


    rgba_image[:, :, :3] = obj  # 复制 RGB 通道
    rgba_image[:, :, 3] = masks[2]  # 将掩码作为 Alpha 通道
    # 裁剪目标区域
    crop_rgba_image = rgba_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]
    img_inpainted = inpaint_img_with_lama(
    img, masks[2], args.lama_config, args.lama_ckpt, device=device)



    return img_inpainted,crop_rgba_image
def remove_and_swap(img, coords1, coords2, bbox1,bbox2,position1,position2,args):
    tmp,obj1 = remove_single_obj(img, coords1, bbox1, args)
    # cv2.imwrite("removed_tmp.png", tmp)
    # cv2.imwrite("obj1.png", obj1)
    res,obj2 = remove_single_obj(tmp, coords2, bbox2, args)
    # cv2.imwrite("obj2.png", obj2)
    # res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("removed.png", res)
    res = Image.fromarray(res,mode='RGB')
    obj1 = Image.fromarray(obj1)
    obj2 = Image.fromarray(obj2)
    new_position1 = bbox2  
    new_position2 = bbox1  
    # 计算目标的偏移量
    # 偏移量 = 目标中心对齐后的左上角坐标 - 目标原始左上角坐标
    offset_x1 = coords2[0] - coords1[0]
    offset_y1 = coords2[1] - coords1[1]
    offset_x2 = coords1[0] - coords2[0]
    offset_y2 = coords1[1] - coords2[1]
    new_position1 = (position1[0] + offset_x1, position1[1] + offset_y1)
    new_position2 = (position2[0] + offset_x2, position2[1] + offset_y2)
    res.paste(obj1, new_position1, obj1)
    res.paste(obj2, new_position2, obj2)

    image_rgb = np.array(res)


    image_bgr = image_rgb[..., ::-1]
    res = Image.fromarray(image_bgr)
    res.save("final.png")
    return res

def refinement(img,bbox1,bbox2,args):
    ...

def swap_in_folder(input_folder, output_folder):
    """
    批量交换文件夹中的目标（如人类对象）位置，并保存修改后的图像。
    
    input_folder: 输入文件夹路径，包含待处理的图像文件
    output_folder: 输出文件夹路径，保存修改后的图像
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = 0
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        try:
            obj1_center, obj2_center, obj1_bbox, obj2_bbox,pos_1,pos_2 = get_clicked_points(image_path)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
        if obj1_center is None:
            logging.warning("No Changes,skip...")
            continue
        logging.info(f"Processing {image_file} with object centers at {obj1_center} and {obj2_center}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image {image_file}")
            continue
        count += 1
        swapped_image = remove_and_swap(image, obj1_center, obj2_center, obj1_bbox, obj2_bbox,pos_1,pos_2, args)
        
        # 保存修改后的图像
        output_image_path = output_folder / image_file
        swapped_image.save(output_image_path)

        print(f"Processed {image_file} and saved to {output_image_path}")
        return count


if __name__ == "__main__":
    args = setup_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # coords1,coords2,bbox1,bbox2,point1,point2 = get_clicked_points(args.input_img)
    # remove_and_swap(args.input_img, coords1,coords2, bbox1,bbox2,point1,point2, args)
    swap_in_folder(args.input_folder, args.output_dir)

