import os
import json
import torch
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama
from utilss import dilate_mask

import logging
from pathlib import Path

# 设置日志记录
logging.basicConfig(level=logging.INFO, 
                    format="%(levelname)s: %(message)s",
                    filename="./log/resize.log",
                    filemode="a"
                    )

# 加载YOLOv5模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO('/root/autodl-tmp/yolov5lu.pt')  # 更新为您模型的路径

# 加载SAM和LaMa模型
sam_checkpoint = "/root/autodl-tmp/sam_vit_h_4b8939.pth" 
lama_ckpt = "/root/autodl-tmp/big-lama"
lama_config = "./lama/configs/prediction/default.yaml"
model_lama = build_lama_model(lama_config, lama_ckpt, device)

# 标签映射
label_mapping = model.names

def set_args(input_img=None, input_folder=None):
    class Args:
        def __init__(self):
            self.input_img = input_img
            self.input_folder = input_folder
            self.output_folder = "./resize_res"
            self.dilate_kernel_size = 7
            self.point_labels = [1]
            self.sam_ckpt = sam_checkpoint
            self.sam_model_type = "vit_h"
            self.lama_config = lama_config
            self.lama_ckpt = lama_ckpt  
    return Args()

def get_mask(img, coords, args):
    masks, _, _ = predict_masks_with_sam(
        img,
        [coords],
        args.point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    ) 
    masks = masks.astype(np.uint8) * 255
    return masks

def get_obj(img, coords, args):
    masks, _, _ = predict_masks_with_sam(
        img,
        [coords],
        args.point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    ) 
    masks = masks.astype(np.uint8) * 255
    rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = img
    rgba_image[:, :, 3] = masks[1]
    return rgba_image, masks

def resize_and_mask_object(img, box, coords_bg, scale_factor, args):
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

    cropped_masks = get_mask(img, coords_bg, args)

    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    resized_img_mask = cv2.resize(cropped_masks[1], (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    scaled_x1 = int(x1 * scale_factor)
    scaled_y1 = int(y1 * scale_factor)
    scaled_x2 = int(x2 * scale_factor)
    scaled_y2 = int(y2 * scale_factor)

    resized_object = resized_img[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
    resized_object_mask = resized_img_mask[scaled_y1:scaled_y2, scaled_x1:scaled_x2]

    rgba_object = np.zeros((resized_object.shape[0], resized_object.shape[1], 4), dtype=np.uint8)
    rgba_object[:, :, :3] = resized_object  # RGB通道
    rgba_object[:, :, 3] = resized_object_mask  # Alpha通道

    return rgba_object, resized_object_mask

def load_pth(pth_path):
    return torch.load(pth_path, map_location='cpu')

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def is_box_within(box1, box2):
    return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]

def resize_img_from_url(img_url, ent_train_dict, caption_dict, args):
    """
    根据图片URL进行处理,返回修改后的图像
    """
    try:
        img = Image.open(img_url).convert("RGB")
        img = np.array(img)
    except Exception as e:
        logging.error(f"无法下载或读取图片 {img_url}: {e}")
        return None


    img_filename = os.path.basename(img_url)
    

    if img_filename not in ent_train_dict or img_filename not in caption_dict:
        logging.warning(f"图片 {img_filename} 不存在于 .pth 或 .json 文件中")
        return None
    
    obj_boxes = ent_train_dict[img_filename]  #{'[OBJ0]': (cx, cy, w, h), ...}
    obj_captions = caption_dict[img_filename]  #{'[OBJ0]': "caption", ...}
    
    # 将相对坐标转换为绝对坐标
    height, width, _ = img.shape
    obj_absolute_boxes = {}
    for obj_id, (cx, cy, w, h) in obj_boxes.items():
        abs_x1 = max(0,int((cx - w/2) * width))
        abs_y1 = max(0,int((cy - h / 2) * height))
        abs_x2 = int((cx + w/2) * width)
        abs_y2 = int((cy + h/2) * height)
        obj_absolute_boxes[obj_id] = (abs_x1, abs_y1, abs_x2, abs_y2)
    

    results = model(img)
    
    Processed = False
    label_processed = set()
    
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            label = box.cls.cpu().item()
            label_name = label_mapping[int(label)]
            
            if conf <= 0.5:
                continue  
            

            if label_name in label_processed:
                continue
            label_processed.add(label_name)
            

            det_x1, det_y1, det_x2, det_y2 = map(int, box.xyxy[0].cpu().numpy())
            det_box = (det_x1, det_y1, det_x2, det_y2)
            
            matched = False
            matched_obj_id = None
            for obj_id, obj_box in obj_absolute_boxes.items():
                if is_box_within(det_box, obj_box):
                    matched = True
                    matched_obj_id = obj_id
                    break
            if not matched:
                continue  
            

            caption = obj_captions.get(matched_obj_id, "")

            if label_name == 'person':
                continue
            

            Processed = True
            scale_factor = np.random.choice([1.2])
            if scale_factor == 1.2:
                logging.info(f"Enlarge object {label_name} in image {img_filename}")
            else:
                logging.info(f"Shrink object {label_name} in image {img_filename}")
            

            center_x_bg = int((det_x1 + det_x2) / 2)
            center_y_bg = int((det_y1 + det_y2) / 2)
            coords_bg = (center_x_bg, center_y_bg)
            

            rgba_object, masks_obj = resize_and_mask_object(img, box, coords_bg, scale_factor, args)
            

            cv2.imwrite("./resize_tmp/mask_obj.jpg", masks_obj)
            rgba_obj = Image.fromarray(rgba_object, mode="RGBA")
            rgba_obj.save("./resize_tmp/rgba_obj.png")
            masks = get_mask(img, coords_bg, args)
            cv2.imwrite("./resize_tmp/mask_0.jpg", masks[1])
            

            if args.dilate_kernel_size is not None:
                masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]
            img_inpainted = inpaint_img_with_builded_lama(
                model_lama, img, masks[1], args.lama_config, device=device
            )
            img = img_inpainted
            cv2.imwrite("./resize_tmp/img_inpat.jpg", img)
            

            bg = Image.fromarray(img, mode="RGB")
            fg = Image.fromarray(rgba_object, mode="RGBA")
            pos = (det_x1, det_y1)
            offset = ((coords_bg[0] - pos[0]) * scale_factor, (coords_bg[1] - pos[1]) * scale_factor)
            new_pos = (int(coords_bg[0] - offset[0]), int(coords_bg[1] - offset[1]))
            
            bg.paste(fg, new_pos, fg)
            img = np.array(bg)
    

    if Processed:
        # resized_img_path = "./resize_tmp/resized_img.jpg"
        resized_img_path = os.path.join(args.output_dir, img_filename)
        cv2.imwrite(resized_img_path, img)
        logging.info(f"Processed image saved to {resized_img_path}")
        return img
    else:
        logging.warning(f"Image {img_filename} not processed")
        return None

def batch_process_images(
    img_folder: str,
    ent_train_dict: str,
    caption_dict: str,
    args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    image_files = [
        os.path.join(img_folder, file) for file in os.listdir(img_folder)
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    for image_path in image_files:
        print(f"Processing image: {image_path}")
        resize_img_from_url(
            image_path=image_path,
            ent_train_dict=ent_train_dict,
            caption_dict=caption_dict,
            args=args
        )


def main():

    args = set_args()
    

    pth_path = "/autodl-fs/data/data/data/MORE/ent_train_dict.pth"  # 更新为实际路径
    json_path = "/autodl-fs/data/data/data/MORE/caption_dict.json"  # 更新为实际路径
    ent_train_dict = load_pth(pth_path)
    caption_dict = load_json(json_path)
    
    # 示例图片URL
    img_url = "/autodl-fs/data/data/data/MORE/img_org/train/3c0ef1fe-cf1c-5465-af18-a8f12c8a3b29.jpg"
    # sftp://root@connect.cqa1.seetacloud.com:29668/autodl-fs/data/data/data/MORE/img_org/train/ff9758cc-659b-5070-ad58-fbcb083a08d2.jpg
    # 处理图片
    resized_img = resize_img_from_url(img_url, ent_train_dict, caption_dict, args)
    
    if resized_img is not None:

        result_image = Image.fromarray(resized_img)
        result_image.show()  
        result_image.save("resized_image.jpg")
    else:
        logging.info("未对图片进行处理。")

if __name__ == "__main__":
    main()
