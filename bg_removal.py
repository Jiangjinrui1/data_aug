import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import ast
import logging

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import fill_img_with_sd,replace_img_with_sd
from lama_inpaint import inpaint_img_with_lama
from utilss.utilss import load_img_to_array, save_array_to_img, dilate_mask
from transformers import CLIPProcessor, CLIPModel
from typing import List
from diffusers import StableDiffusionInpaintPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
    ).to(device)
lama_ckpt = "/root/autodl-tmp/big-lama"
lama_config = "./lama/configs/prediction/default.yaml"

def load_txt_data(txt_file: str):
    """
    从 txt 文件中读取数据，按 img_id 组织并提取每个对象的 pos 信息
    """
    txt_data = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = ast.literal_eval(line)
            img_id = line_data['img_id']
            obj_name = line_data.get('t', {}).get('name')  # 获取对象名称，例如 '[OBJ0]'

            if not obj_name:
                continue
            
            # 初始化 img_id 键
            if img_id not in txt_data:
                txt_data[img_id] = {}
            
            # 将对象的数据保存到指定 img_id 的对象键下
            txt_data[img_id][obj_name] = {
                "text": line_data['text'],
                "token": line_data['token'],
                "h": line_data['h'],
                "relation": line_data['relation'],
                "pos": line_data['t'].get('pos')  # 提取 pos 信息
            }
    return txt_data

def process_single_object(
    input_img: str,
    text_prompt: str,
    original_prompt: str,
    output_dir: str,
    obj_box: tuple,
    sam_model_type: str = 'vit_h',
    sam_ckpt: str = 'sam_vit_h_4b8939.pth',
    operation: str = 'remove',
    seed: int = None,
    dilate_kernel_size = 15
):
    """处理单个 OBJ 实体的图像，并生成带遮罩的图像"""
    logging.info(f"Processing image {input_img} for object with bounding box {obj_box}")
    modified_caption = text_prompt

    img = load_img_to_array(input_img)
    img_height, img_width = img.shape[:2]
    
    # 根据原始图像尺寸，将相对坐标转换为绝对坐标
    # abs_box = (
    #     int(obj_box[0] * img_width),
    #     int(obj_box[1] * img_height),
    #     int(obj_box[2] * img_width),
    #     int(obj_box[3] * img_height)
    # )
    
    # 获取 OBJ 中心坐标作为 SAM 的点击点
    # center_x = (abs_box[0] + abs_box[2]) // 2
    # center_y = (abs_box[1] + abs_box[3]) // 2
    center_x, center_y, box_width, box_height = obj_box
    # 将相对坐标转换为绝对坐标
    center_x = int(center_x * img_width)
    center_y = int(center_y * img_height)
    coords = [center_x, center_y]
    
    # 使用 SAM 模型生成遮罩
    masks, _, _ = predict_masks_with_sam(
        img, [coords], [1],
        model_type=sam_model_type, ckpt_p=sam_ckpt, device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    masks = masks.astype(np.uint8) * 255
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    # 设置输出目录
    img_stem = Path(input_img).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_pairs = []
    if operation == "remove":
        if seed is not None:
            torch.manual_seed(seed)
        img_removed = inpaint_img_with_lama(img, masks[2], config_p = lama_config, ckpt_p= lama_ckpt, device=device)
        # candidate_pairs.append((img,img_removed))
        # best_pair = select_best_image_pair(candidate_pairs,original_prompt,modified_caption)
        save_path = out_dir / f"{img_stem}.jpg"
        save_array_to_img(img_removed, save_path) 
        print(f"saved in {save_path}")             


def pipeline_process_images_from_folder(
    img_folder: str,
    caption_file: str,
    cap_modified_file: str,
    output_dir: str,
    txt_file: str,
    dilate_kernel_size: int = None,
    sam_model_type: str = 'vit_h',
    sam_ckpt: str = 'sam_vit_h_4b8939.pth',
    seed: int = None
):
    """批量处理文件夹中的图像并根据caption生成prompt"""
    # 创建输出目录
    """批量处理文件夹中的图像并根据caption生成prompt"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载 JSON 文件内容
    with open(caption_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    with open(cap_modified_file, 'r', encoding='utf-8') as f:
        counterfactual_data = json.load(f)
    txt_data = load_txt_data(txt_file)
    processed_img = set()
    building_keywords = [
        "building", "structure", "tower", "skyscraper", "house", "apartment",
        "office", "block", "villa", "mansion", "condominium", "complex",
        "residence", "hotel", "lodge", "inn", "cabin", "bungalow", "shack",
        "shed", "barn", "factory", "warehouse", "facility", "dormitory",
        "hospital", "school", "church", "temple", "mosque", "synagogue",
        "chapel", "castle", "palace", "monument", "dome", "hall", "station",
        "headquarters", "quarters", "embassy", "center", "stadium", "arena",
        "gym", "court", "gallery", "museum", "theater", "cinema", "mall",
        "shopping center", "plaza", "market", "library", "pavilion", "fort"
    ]

    # 处理每张图像
    for img_id, objs in counterfactual_data.items():
        if img_id not in original_data:
            logging.warning(f"Image {img_id} not found in original caption data, skipping.")
            continue
        if img_id not in txt_data:
            logging.warning(f"Image {img_id} not found in txt data, skipping.")
            continue
        if img_id in processed_img:
            logging.info(f"Image {img_id} has been processed, skipping.")
            continue
        # 图像路径
        img_path = os.path.join(img_folder, img_id)

        img_folder_name = os.path.splitext(img_id)[0]
        if os.path.exists(os.path.join(output_dir, img_folder_name)):
            logging.info(f"Image {img_id} already exists in output folder, skipping.")
            continue
        if not os.path.exists(img_path):
            logging.warning(f"Image {img_id} not found in folder {img_folder}, skipping.")
            continue

        # 获取每个对象的边框信息并找到面积最大的对象
        max_area = 0
        largest_obj_id = None
        largest_obj_box = None
        largest_original_caption = ""
        largest_modified_caption = ""


        for obj_id, modified_caption in objs.items():
            if obj_id not in txt_data[img_id]:
                logging.warning(f"Object {obj_id} not found for image {img_id} in txt data, skipping.")
                continue

            # 获取对象的边框坐标
            obj_box = txt_data[img_id][obj_id]['pos']
            original_caption = original_data[img_id].get(obj_id, "")

            if any(keyword in modified_caption.lower() for keyword in building_keywords):
                logging.info(f"Building object found in image {img_id}, selecting object {obj_id} for processing.")
                largest_obj_id = obj_id
                largest_obj_box = obj_box
                largest_original_caption = original_caption
                largest_modified_caption = modified_caption
                operation = "remove"  # 如果包含建筑物关键词，则设置操作为删除

                process_single_object(
                input_img=img_path,
                text_prompt=largest_modified_caption,
                original_prompt=largest_original_caption,
                output_dir=output_dir,
                obj_box=largest_obj_box,
                sam_model_type=sam_model_type,
                sam_ckpt=sam_ckpt,
                operation=operation,
                seed=seed
                )
                largest_obj_id = None
                break


if __name__ == "__main__":
    img_folder = r"/autodl-fs/data/data/data/MORE/img_org/train"
    txt_file = r"/autodl-fs/data/data/data/MORE/txt/train.txt"
    caption_file = r"/autodl-fs/data/data/data/MORE/caption_dict.json"
    # sftp://root@connect.cqa1.seetacloud.com:29668/autodl-fs/data/data/data/MORE/caption_dict.json
    cap_modified_file = r"/autodl-fs/data/data/data/MORE/caption_modified.json"  # 修改后的 caption 文件
    output_dir = '/root/autodl-fs/data/results_more/train'
    
    dilate_kernel_size = 15
    sam_model_type = 'vit_h'
    sam_ckpt = '/root/autodl-tmp/sam_vit_h_4b8939.pth'
    seed = None
    
    pipeline_process_images_from_folder(
        img_folder=img_folder,
        txt_file=txt_file,
        caption_file=caption_file,
        cap_modified_file=cap_modified_file,
        output_dir=output_dir,
        dilate_kernel_size=dilate_kernel_size,
        sam_model_type=sam_model_type,
        sam_ckpt=sam_ckpt,
        seed=seed
    )
