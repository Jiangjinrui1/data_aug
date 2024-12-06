import json
import torch
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from diffusers import StableDiffusionInpaintPipeline
from ultralytics import YOLO
from stable_diffusion_inpaint import fill_img_with_sd
from sam_segment import predict_masks_with_sam
from utilss import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_yolo = YOLO('/root/autodl-tmp/yolov5lu.pt')
sam_model_type = "vit_h"
sam_ckpt = "/root/autodl-tmp/sam_vit_h_4b8939.pth"
outdir = "./output"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
    ).to(device)

def setup_args():
    class Args:
        # input_img: str = "./example/replace-anything/dog.png"
        input_img: str = "/autodl-fs/data/data/data/MORE/img_org/total/bbde4e6c-9cad-50d8-b860-12cb800357f7.jpg"
        coords_type: str = "key_in"
        text_prompt: str = "a man with pink clothes"
        dilate_kernel_size: int = 15
        output_dir: str = "./results"
        sam_model_type: str = "vit_h"
        sam_ckpt: str = "/root/autodl-tmp/sam_vit_h_4b8939.pth"
        seed: int = None
        deterministic: bool = False
    return Args()
def load_ent_train_dict(pth_path):
    """
    加载 ent_train_dict.pth 文件，返回一个字典。
    """
    ent_dict = torch.load(pth_path, map_location='cpu')
    return ent_dict

def load_caption_dict(json_path):
    """
    加载 caption_dict.json 或 caption_dict_modified.json 文件，返回一个字典。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        caption_dict = json.load(f)
    return caption_dict

def load_txt_relations(txt_path):
    """
    加载 txt 文件，解析每行的 JSON 字符串，返回一个字典。
    格式：{img_id: {obj_id: relation, ...}, ...}
    """
    relations_dict = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            img_id = data['img_id']
            obj_id = data['t']['name']
            relation = data.get('relation', 'none')
            if img_id not in relations_dict:
                relations_dict[img_id] = {}
            relations_dict[img_id][obj_id] = relation
    return relations_dict

def filter_objects(ent_train_dict, relations_dict):
    """
    过滤具有特定关系的对象，返回一个新的字典。
    """
    filtered_dict = {}
    for img_id, objs in ent_train_dict.items():
        if img_id not in relations_dict:
            filtered_dict[img_id] = objs
            continue
        filtered_objs = {}
        for obj_id, bbox in objs.items():
            relation = relations_dict[img_id].get(obj_id, 'none')
            if relation == 'none':
                filtered_objs[obj_id] = bbox
        if filtered_objs:
            filtered_dict[img_id] = filtered_objs
    return filtered_dict
def detect_and_match_objects(image, model_yolo, filtered_ent_train, img_id):
    """
    使用YOLOv5检测图像中的对象，并匹配数据集中的OBJ框。
    返回匹配的检测结果列表。
    """
    results = model_yolo(image)
    matched_objects = []
    if img_id not in filtered_ent_train:
        return matched_objects
    
    dataset_objs = filtered_ent_train[img_id]
    for r in results:
        for box in r.boxes:
            label = int(box.cls[0])
            label_name = model_yolo.names[label]
            confidence = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_bbox = (x1, y1, x2, y2)
            
            for obj_id, (cx, cy, w, h) in dataset_objs.items():
                img_h, img_w = image.shape[:2]
                abs_x1 = max(0,int((cx - w/2) * img_w))
                abs_y1 = max(0,int((cy - h / 2) * img_h))
                abs_x2 = int((cx + w/2) * img_w)
                abs_y2 = int((cy + h/2) * img_h)
                obj_box = (abs_x1, abs_y1, abs_x2, abs_y2)
                if box_within(detected_bbox, obj_box):
                    matched_objects.append({
                        "label": label_name,
                        "confidence": confidence,
                        "bbox": detected_bbox,
                        "obj_id": obj_id,
                        "dataset_bbox": obj_box
                    })
                    break  # 一个检测框只匹配一个OBJ
    return matched_objects

def box_within(inner_box, outer_box):
    """
    检查 inner_box 是否完全包含在 outer_box 内。
    """
    return (inner_box[0] >= outer_box[0] and
            inner_box[1] >= outer_box[1] and
            inner_box[2] <= outer_box[2] and
            inner_box[3] <= outer_box[3])


def fill_anything(image, bbox, text_prompt, pipe, device, seed=None,dilate_kernel_size=5,args = None):
    """
    根据给定的文本提示修改图像中指定区域的属性。
    
    Args:
        image (numpy.ndarray): 原始图像。
        bbox (tuple): 对象的边界框 (x1, y1, x2, y2)。
        text_prompt (str): 修改属性的文本提示。
        pipe (StableDiffusionInpaintPipeline): Stable Diffusion Inpainting 管道。
        device (str): 设备类型，如 'cuda' 或 'cpu'。
        seed (int, optional): 随机种子。
    
    Returns:
        numpy.ndarray: 修改后的图像。
    """
    img = image.copy()

    results = model_yolo(img)
    latest_coords = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cls = int(box.cls[0])
            center = ((x1+x2)/2, (y1+y2)/2)
            latest_coords = center
            break
        

    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        point_labels = [1],
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # fill the masked image
    imgs_filled = []
    for idx, mask in enumerate(masks):
        if seed is not None:
            torch.manual_seed(seed)
        img_filled = fill_img_with_sd(
            img, mask, text_prompt, device=device,pipe=pipe)
        imgs_filled.append(img_filled)
    return imgs_filled


def process_image(
    image_path: str,
    ent_train_pth: str,
    caption_json: str,
    caption_modified_json: str,
    txt_path: str,
    model_yolo: YOLO,
    pipe: StableDiffusionInpaintPipeline,
    device: str,
    sam_model_type: str,
    sam_ckpt: str,
    seed: int = None,
    args = None
):
    """
    处理单张图像，实现基于 relation 为 None 的对象属性修改。
    
    Args:
        image_path (str): 输入图像路径。
        ent_train_pth (str): ent_train_dict.pth 文件路径。
        caption_json (str): caption_dict.json 文件路径。
        caption_modified_json (str): caption_dict_modified.json 文件路径。
        txt_path (str): txt 文件路径。
        model_yolo (YOLO): YOLOv5 模型实例。
        pipe (StableDiffusionInpaintPipeline): Stable Diffusion Inpainting 管道。
        device (str): 设备类型，如 'cuda' 或 'cpu'。
        sam_model_type (str): SAM 模型类型。
        sam_ckpt (str): SAM 模型权重路径。
        seed (int, optional): 随机种子。
    
    Returns:
        None
    """
    # 加载数据文件
    ent_train_dict = load_ent_train_dict(ent_train_pth)
    caption_dict = load_caption_dict(caption_json)
    caption_modified_dict = load_caption_dict(caption_modified_json)
    relations_dict = load_txt_relations(txt_path)
    
    # 过滤具有 relation 为 'none' 的对象
    filtered_ent_train = filter_objects(ent_train_dict, relations_dict)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取 img_id
    img_id = os.path.basename(image_path)
    
    # 检测并匹配对象
    matched_objects = detect_and_match_objects(image_rgb, model_yolo, filtered_ent_train, img_id)
    
    if not matched_objects:
        print("No matching objects found with relation 'none'.")
        return  # 返回原图
    
    for obj in matched_objects:
        obj_id = obj["obj_id"]
        bbox = obj["bbox"]
        modified_caption = caption_modified_dict.get(img_id, {}).get(obj_id, "")
        original_caption = caption_dict.get(img_id, {}).get(obj_id, "")
        if not modified_caption:
            print(f"No modified caption found for object {obj_id} in image {img_id}. Skipping.")
            continue
        # 检查obj占原图的比例，若小于10%，设置dilate_kernel_size为 3
        if bbox[2] * bbox[3] / (image.shape[0] * image.shape[1]) < 0.1:
            dilate_kernel_size = 3
        elif bbox[2] * bbox[3] / (image.shape[0] * image.shape[1]) < 0.15:
            dilate_kernel_size = 5
        elif bbox[2] * bbox[3] / (image.shape[0] * image.shape[1]) < 0.2:
            dilate_kernel_size = 7
        else:
            dilate_kernel_size = 15
        # 使用 Stable Diffusion 进行 inpainting
        images = fill_anything(
            image=image,
            bbox=bbox,
            text_prompt=modified_caption,
            pipe=pipe,
            device=device,
            seed=seed,
            dilate_kernel_size=dilate_kernel_size,
            args=args,
        )
    
    for i, img in enumerate(images):
        # 获取输入图像的文件夹路径
        input_dir = os.path.dirname(image_path)
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(input_dir, img_basename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(output_path, img)
        # 输出保存路径
        print(f"Saved modified image to {output_path}")

def main():
    args = setup_args()
    image_path = "/autodl-fs/data/data/data/MORE/img_org/total/bbde4e6c-9cad-50d8-b860-12cb800357f7.jpg"
    url_text = r"/autodl-fs/data/data/data/MORE/txt/train.txt"
    url_json = r"/autodl-fs/data/data/data/MORE/caption_dict.json"
    url_mod_json = r"/autodl-fs/data/data/data/MORE/caption_modified_dict.json"
    url_pth = r"/autodl-fs/data/data/data/MORE/ent_train_dict.pth" 
    process_image(image_path, 
                  url_pth, 
                  url_json,
                  url_mod_json,
                  url_text,
                  model_yolo,
                  pipe,
                  device,sam_model_type,sam_ckpt,
                  seed=None,
                  args=args)


if __name__ == "__main__":
    main()