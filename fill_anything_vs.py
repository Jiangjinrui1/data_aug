import sys
sys.path.append('.')
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import cv2
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import fill_img_with_sd,replace_img_with_sd
from utilss.utilss import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
from typing import Any, Dict, List
from facenet_pytorch import MTCNN
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

import ast,json
from openai import OpenAI

import logging
logging.basicConfig(
    filename='process_log.log',  # 日志文件名称
    filemode='a',  # 追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 控制台输出日志
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载预训练的YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
# model_v7 = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='yolov7.pt')
# 加载 MTCNN 面部检测模型
mtcnn = MTCNN()

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
    ).to(device)


# 加载 CLIP 模型
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def person_detect(img,min_face_area_ratio=0.05):
    img = np.array(img)
    img_height, img_width = img.shape[:2]
    min_face_area = img_height * img_width * min_face_area_ratio 
    results = model(img)
    results.render()  # 可视化结果
    detections = results.pandas().xyxy[0]
    person_detections = detections[detections['name'] == 'person']
    if len(person_detections) > 0:
        # 使用 MTCNN 检测人脸
        boxes, _ = mtcnn.detect(img)
        
        # 如果检测到人脸，但面积较小则返回 person 框中心
        if boxes is not None:
            for box in boxes:
                xmin, ymin, xmax, ymax = map(int, box)
                face_area = (xmax - xmin) * (ymax - ymin)
                
                if face_area >= min_face_area:
                    center_x = (xmin + xmax) // 2
                    center_y = (ymin + ymax) // 2
                    logging.info(f"Face detected with sufficient area at ({center_x}, {center_y}), proceeding with face replacement.")
                    tag = 'face'
                    return center_x, center_y,tag
                else:
                    logging.info("Face detected, but too small for effective replacement. Proceeding with person bounding box replacement.")
        
        # 若未检测到足够大的人脸，返回第一个 person 框的中心点
        person_box = person_detections.iloc[0]
        xmin, ymin, xmax, ymax = int(person_box['xmin']), int(person_box['ymin']), int(person_box['xmax']), int(person_box['ymax'])
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        logging.info(f"Replacing entity in person bounding box at center: ({center_x}, {center_y})")
        tag = 'person'
        return center_x, center_y,tag
    else:
        logging.info("No person detected in image.")
        return None
def detect_largest_object(results):
    """找到YOLO检测中面积最大的对象"""
    detections = results.pandas().xyxy[0]
    if detections.empty:
        logging.info("no object detected")
        return None
    # 计算面积 (width * height)
    detections['area'] = (detections['xmax'] - detections['xmin']) * (detections['ymax'] - detections['ymin'])
    # 找到面积最大的对象
    largest_object = detections.loc[detections['area'].idxmax()]
    return largest_object
def expand_box(box, img_width, img_height, factor=1.2):
    """
    扩大检测框的大小。

    Args:
        box (tuple): 原始检测框 (xmin, ymin, xmax, ymax)。
        img_width (int): 图像宽度。
        img_height (int): 图像高度。
        factor (float): 扩大倍数，默认 1.2。

    Returns:
        tuple: 扩大后的检测框 (xmin, ymin, xmax, ymax)。
    """
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    center_x, center_y = xmin + w // 2, ymin + h // 2

    new_w, new_h = int(w * factor), int(h * factor)
    new_xmin = max(center_x - new_w // 2, 0)
    new_ymin = max(center_y - new_h // 2, 0)
    new_xmax = min(center_x + new_w // 2, img_width)
    new_ymax = min(center_y + new_h // 2, img_height)

    return new_xmin, new_ymin, new_xmax, new_ymax
def process_image_with_sam(
    input_img: str,
    coords_type: str,
    point_coords: List[float],
    point_labels: List[int],
    text_prompt: str,
    dilate_kernel_size: int = None,
    output_dir: str = './results',
    sam_model_type: str = 'vit_h',
    sam_ckpt: str = 'sam_vit_h_4b8939.pth',
    seed: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """处理图像并生成 mask 和修复结果。

    Args:
        input_img (str): 输入图像路径
        coords_type (str): 坐标选择方式，"click" 或 "key_in"
        point_coords (List[float]): 手动输入的坐标点
        point_labels (List[int]): 坐标标签，1 或 0
        text_prompt (str): 文本提示，描述要生成的图像内容
        dilate_kernel_size (int, optional): 膨胀掩码大小
        output_dir (str): 输出文件夹路径
        sam_model_type (str): SAM 模型类型
        sam_ckpt (str): SAM 模型的权重文件路径
        seed (int, optional): 随机种子
        device (str): 设备，"cuda" 或 "cpu"
    """
    
    # 加载图像
    img = load_img_to_array(input_img)
    
    # 选择坐标方式
    if coords_type == "click":
        latest_coords = get_clicked_point(input_img)  # 实现此函数从图像中获取点击点
    elif coords_type == "key_in":
        latest_coords = point_coords
    
    # 使用 SAM 模型预测掩码
    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        point_labels,
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )
    
    masks = masks.astype(np.uint8) * 255  # 转换掩码

    # 如果设置了膨胀掩码，则对掩码进行膨胀操作
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # 设置输出目录
    img_stem = Path(input_img).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 可视化和保存掩码及结果
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        save_array_to_img(mask, mask_p)

        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], point_labels, size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 使用 Stable Diffusion 修复被掩码的图像
    for idx, mask in enumerate(masks):
        if seed is not None:
            torch.manual_seed(seed)
        img_filled_p = out_dir / f"filled_with_mask_{idx}.png"
        img_filled = fill_img_with_sd(img, mask, text_prompt, device=device,pipe = pipe)
        save_array_to_img(img_filled, img_filled_p)

    print(f"Processing complete. Results saved to: {out_dir}")

def process_image_with_replacement(
    input_img: str,
    coords_type: str,
    point_coords: List[float],
    point_labels: List[int],
    text_prompt: str,
    dilate_kernel_size: int = None,
    output_dir: str = './results',
    sam_model_type: str = 'vit_h',
    sam_ckpt: str = 'sam_vit_h_4b8939.pth',
    seed: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """处理图像并生成 mask 和替换结果。

    Args:
        input_img (str): 输入图像路径
        coords_type (str): 坐标选择方式，"click" 或 "key_in"
        point_coords (List[float]): 手动输入的坐标点
        point_labels (List[int]): 坐标标签，1 或 0
        text_prompt (str): 文本提示，描述要生成的图像内容
        dilate_kernel_size (int, optional): 膨胀掩码大小
        output_dir (str): 输出文件夹路径
        sam_model_type (str): SAM 模型类型
        sam_ckpt (str): SAM 模型的权重文件路径
        seed (int, optional): 随机种子
        device (str): 设备，"cuda" 或 "cpu"
    """
    
    # 加载图像
    img = load_img_to_array(input_img)
    
    # 选择坐标方式
    if coords_type == "click":
        latest_coords = get_clicked_point(input_img)  # 实现此函数从图像中获取点击点
    elif coords_type == "key_in":
        latest_coords = point_coords
    
    # 使用 SAM 模型预测掩码
    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        point_labels,
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )
    
    masks = masks.astype(np.uint8) * 255  # 转换掩码

    # 如果设置了膨胀掩码，则对掩码进行膨胀操作
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # 设置输出目录
    img_stem = Path(input_img).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 可视化和保存掩码及结果
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        save_array_to_img(mask, mask_p)

        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], point_labels, size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 使用 Stable Diffusion 替换被掩码的图像
    for idx, mask in enumerate(masks):
        if seed is not None:
            torch.manual_seed(seed)
        img_replaced_p = out_dir / f"replaced_with_mask_{idx}.png"
        img_replaced = replace_img_with_sd(img, mask, text_prompt, device=device)
        save_array_to_img(img_replaced, img_replaced_p)

    print(f"Processing complete for {input_img}. Results saved to: {out_dir}")
def generate_face_edit_prompt(client, face_description, model="gpt-4o-mini"):
    """
    生成简洁、固定格式的面部及头部细节修复 prompt。

    Args:
        client (object): OpenAI API 客户端实例。
        face_description (str): 面部及头部特征的描述。
        model (str): 使用的 OpenAI 模型，默认为 "gpt-4"。

    Returns:
        str: 简洁、直接的面部及头部修改 prompt。
    """
    prompt = f"""
    You are an assistant that generates short, descriptive sentences for facial and head features in images.
    Based on the provided description, create a sentence that briefly captures the person's facial expression, one notable facial feature, and a detail about the head (like hairstyle or accessories).

    Format: "[expression] with [facial feature], [head detail]"
    
    Description: "{face_description}"

    Generate a descriptive sentence in the specified format.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Generate a concise, descriptive sentence focused on facial and head features, in the format '[expression] with [facial feature], [head detail]'."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=20  # 稍微增加长度以包含头部细节
    )
    
    return response.choices[0].message.content.strip()
def generate_contextual_background_prompt(client, sentence, model="gpt-4o-mini"):
    """
    根据给定句子内容生成背景修改的简洁 prompt。

    Args:
        client (object): OpenAI API 客户端实例。
        sentence (str): 修改后的句子，描述新的场景或内容。
        model (str): 使用的 OpenAI 模型，默认为 "gpt-4"。

    Returns:
        str: 针对背景修改的 prompt。
    """
    prompt = f"""
    You are an assistant that generates background modification prompts for images.
    You will be provided with a sentence that describes a scene or action.
    
    Your task is to infer an appropriate background based on the scene or activity in the sentence and generate a brief prompt describing this new background setting.
    - Avoid restating specific entities or actions in the sentence.
    - Focus on providing a setting or location that complements the activity or scene described.
    - Keep the description concise and relevant to the scene, such as "a sunny park" or "a bustling city street."

    Sentence: "{sentence}"

    Based on this scene, provide a short background prompt that would fit this activity or setting.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Generate a concise background prompt based on the scene or activity in the sentence."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=20  # 限制生成长度，确保简洁
    )
    
    return response.choices[0].message.content.strip()
def generate_summarized_counterfactual_prompt(client, original_sentence, modified_sentence):
    """生成反事实句子的缩句"""
    prompt = f"""
    You are an assistant that generates concise summaries for counterfactual sentences. 
    You will be provided with two sentences: 
    1. The original sentence that describes an image.
    2. A modified sentence that describes the same image but includes some differences.

    Your task is to generate a brief summary of the modified sentence, highlighting the key differences from the original sentence.
    - Focus on concisely describing the most important changes.
    - Omit minor details and use simple language.

    Original sentence: "{original_sentence}"
    Modified sentence: "{modified_sentence}"

    Based on these two sentences, provide a concise summary of the modified sentence.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate a concise summary for a counterfactual sentence by focusing on key differences."},
            {"role": "user", "content": prompt}
        ],
        max_tokens = 15
    )
    return response.choices[0].message.content
def generate_single_entity_prompts(client, original_sentence):
    """生成仅包含单一实体的候选 prompt，以替换主语。"""
    prompt_request = f"""
    You are an assistant that generates concise, single-entity replacement prompts.
    Based on the sentence: "{original_sentence}", create several options, each describing a single new entity replacing the main subject.
    Each option should be formatted as "a [single entity] in/on [someplace]" and focus on describing one object or person in a natural setting.
    Avoid using groups or multiple entities in the prompt.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_request}],
        max_tokens = 25
    )
    prompts = response.choices[0].message.content.split("\n")
    return [p.strip() for p in prompts if p.strip()]

def select_best_prompt(original_sentence, candidates, clip_model, clip_processor):
    """
    从多个候选 prompt 中选择与原始句子最相似的 prompt。

    Args:
        original_sentence (str): 原始句子。
        candidates (list): 候选 prompt 列表。
        clip_model (object): CLIP 模型实例。
        clip_processor (object): CLIP 处理器实例。

    Returns:
        str: 最佳 prompt。
    """
    inputs = clip_processor(text=[original_sentence] + candidates, return_tensors="pt", padding=True)
    outputs = clip_model.get_text_features(**inputs)

    # 原始句子和候选 prompt 的嵌入
    original_embedding = outputs[0]
    candidate_embeddings = outputs[1:]

    # 计算原始句子与每个候选 prompt 的余弦相似度
    similarities = torch.nn.functional.cosine_similarity(original_embedding.unsqueeze(0), candidate_embeddings)
    best_index = similarities.argmax().item()

    return candidates[best_index]

# 在图像处理流程中使用新函数
def generate_entity_replacement_prompt(client,original_sentence, clip_model, clip_processor):
    # 生成多个候选 prompt
    candidate_prompts = generate_single_entity_prompts(client, original_sentence)

    # 选择与原始句子最相似的 prompt
    best_prompt = select_best_prompt(original_sentence, candidate_prompts, clip_model, clip_processor)

    print(f"Best prompt selected for replacement: {best_prompt}")
    return best_prompt

# 函数：计算图像和文本的相似度
def calculate_clip_similarity(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
    return similarity.item()

# 函数：计算图像嵌入
def get_image_embeddings(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs

# 函数：计算图像对之间的相似度
def calculate_image_similarity(image1, image2):
    image1_embeds = get_image_embeddings(image1)
    image2_embeds = get_image_embeddings(image2)
    
    similarity = torch.nn.functional.cosine_similarity(image1_embeds, image2_embeds, dim=-1)
    return similarity.item()

# 函数：选择 CLIPdir 得分最高的图像对
def select_best_image_pair(image_pairs, original_prompt, counterfactual_prompt):
    best_pair = None
    best_clipdir_score = -float('inf')
    
    for idx, (original_image, modified_image) in enumerate(image_pairs):
        if modified_image is None:
            continue
        original_image =  Image.fromarray(original_image.astype(np.uint8))
        modified_image = Image.fromarray(modified_image.astype(np.uint8))
        
        sim_original = calculate_clip_similarity(original_image, original_prompt)
        sim_modified = calculate_clip_similarity(modified_image, counterfactual_prompt)
        sim_image_pair = calculate_image_similarity(original_image, modified_image)
        
        # 计算 CLIPdir 分数
        directional_similarity = (sim_image_pair + sim_original + sim_modified) / 3
        # 记录中间相似度
        logging.info(
            f"Image pair {idx}: sim_original={sim_original}, sim_modified={sim_modified}, "
            f"sim_image_pair={sim_image_pair}, CLIPdir_score={directional_similarity}"
        )
        
        if directional_similarity > best_clipdir_score:
            best_clipdir_score = directional_similarity
            best_pair = modified_image
    logging.info(f"Best CLIPdir score: {best_clipdir_score}")
    
    return best_pair

def pipeline_process_images_from_folder(
    img_folder: str,
    caption_file: str,
    cap_modified_file: str,
    output_dir: str,
    dilate_kernel_size: int = None,
    sam_model_type: str = 'vit_h',
    sam_ckpt: str = 'sam_vit_h_4b8939.pth',
    seed: int = None,
    client=None  # GPT 客户端
):
    """批量处理文件夹中的图像并根据caption生成prompt"""
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    processed_img = set()

    with open(caption_file, 'r') as orig_f, open(cap_modified_file, 'r') as counter_f:
        for orig_line, counter_line in zip(orig_f, counter_f):
            original_data = ast.literal_eval(orig_line)  # 使用 ast.literal_eval 替代 json.loads
            counterfactual_data = ast.literal_eval(counter_line)
            
            # 提取原始标题和反事实标题
            original_title = original_data['caption']
            counterfactual_title = counterfactual_data['caption']
            img_id = counterfactual_data['img_id']
            img_folder_name = os.path.splitext(img_id)[0]
            # 如果文件夹已经存在，则跳过
            if os.path.exists(os.path.join(output_dir, img_folder_name)):
                logging.info(f"Output folder for image {img_id} already exists, skipping.")
                continue
            if img_id in processed_img:
                logging.info(f"Image {img_id} already processed, skipping.")
                continue
            elif original_title == counterfactual_title:
                logging.info(f"No changes detected for image {img_id}, skipping.")
                continue
            else:
                processed_img.add(img_id)
            # # 构建prompt
            # prompt = generate_summarized_counterfactual_prompt(client, original_title, counterfactual_title)
            # prompt = counterfactual_title
            # logging.info(f"Generated prompt for image {img_id}: {prompt}")
        
            # 图像路径
            img_path = os.path.join(img_folder, img_id)
            if not os.path.exists(img_path):
                logging.warning(f"Image {img_id} not found in folder {img_folder}, skipping.")
                continue
        
            # 处理每张图像
            process_single_image(input_img = img_path, 
                                 text_prompt = counterfactual_title, 
                                 original_prompt = original_title,
                                 output_dir = output_dir, 
                                 sam_model_type = sam_model_type, 
                                 sam_ckpt = sam_ckpt, 
                                 seed = seed
                                 )

def process_single_image(
    input_img: str,
    text_prompt: str,
    original_prompt: str,
    output_dir: str,
    sam_model_type: str = 'vit_h',
    sam_ckpt: str = 'sam_vit_h_4b8939.pth',
    seed: int = None
):
    """处理单张图像，筛选生成的最佳图像"""
    # img = Image.open(input_img)
    # img.show()
    """处理单张图像，筛选生成的最佳图像"""
    logging.info(f"Processing image: {input_img}")
    img_PIL = Image.open(input_img)
    img = load_img_to_array(input_img)
    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width
    ori_sentence = original_prompt
    mod_sentence = text_prompt
    
    # YOLOv5目标检测
    results = model(img)
    
    # 检测person并获取face坐标
    face_coords = person_detect(img)
    
    # 设置输出路径
    img_stem = Path(input_img).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image_pairs = []

    if face_coords is not None and face_coords[2] =='face':
        text_prompt = generate_face_edit_prompt(client,text_prompt,'gpt-4o-mini')
        logging.info(f"Processing image: {input_img} with prompt: {text_prompt}")
        # 替换人脸
        logging.info(f"Person detected at coordinates: {face_coords}, replacing face.")
        # face_coords取得 tuple中前两个元素
        face_coords = face_coords[:2]
        
        masks, _, _ = predict_masks_with_sam(
            img, [face_coords], [1],
            model_type=sam_model_type, ckpt_p=sam_ckpt, device=device
        )
        masks = masks.astype(np.uint8) * 255
        
        for idx, mask in enumerate(masks):
            if seed is not None:
                torch.manual_seed(seed)
            img_filled = fill_img_with_sd(img, mask, text_prompt, device=device, pipe=pipe)
            if img_filled is None:
                logging.warning(f"Failed to generate filled image for face replacement.")
                continue
            image_pairs.append((img, img_filled))
            logging.info(f"Generated {len(image_pairs)} filled images for face replacement.")
    else:    
        largest_object = detect_largest_object(results)
        if largest_object is not None:
            object_area = (largest_object['xmax'] - largest_object['xmin']) * (largest_object['ymax'] - largest_object['ymin'])
            object_area_ratio = object_area / img_area
                        # 检查对象面积比例是否小于 20%
            if object_area_ratio < 0.2:
                # 替换目标框内内容，但不改变背景
                logging.info("Object area ratio is less than 20%, replacing content within the object.")
                text_prompt = generate_entity_replacement_prompt(client, original_prompt,clip_model,clip_processor)
                logging.info(f"Processing image: {input_img} with prompt: {text_prompt}")
                center_x = (largest_object['xmin'] + largest_object['xmax']) // 2
                center_y = (largest_object['ymin'] + largest_object['ymax']) // 2
                coords = [center_x, center_y]

                masks, _, _ = predict_masks_with_sam(
                    img, [coords], [1],
                    model_type=sam_model_type, ckpt_p=sam_ckpt, device=device
                )
                masks = masks.astype(np.uint8) * 255

                for idx, mask in enumerate(masks):
                    if seed is not None:
                        torch.manual_seed(seed)
                    img_filled = fill_img_with_sd(img, mask, text_prompt, device=device, pipe=pipe)
                    image_pairs.append((img, img_filled))
                logging.info(f"Generated {len(image_pairs)} filled images for entity replacement.")
            else:
                text_prompt = generate_contextual_background_prompt(client,text_prompt)
                logging.info(f"Processing image: {input_img} with prompt: {text_prompt}")
                # 如果没有检测到person，找到最大对象并替换背景
                logging.info("No person detected, replacing background of the largest object.")    
                center_x = (largest_object['xmin'] + largest_object['xmax']) // 2
                center_y = (largest_object['ymin'] + largest_object['ymax']) // 2
                coords = [center_x, center_y]

                masks, _, _ = predict_masks_with_sam(
                    img, [coords], [1],
                    model_type=sam_model_type, ckpt_p=sam_ckpt, device=device
                )
                masks = masks.astype(np.uint8) * 255

                for idx, mask in enumerate(masks):
                    if seed is not None:
                        torch.manual_seed(seed)
                    img_replaced = replace_img_with_sd(img, mask, text_prompt, device=device,pipe=pipe)
                    # img_replaced = Image.fromarray(img_replaced)
                    image_pairs.append((img, img_replaced))
                logging.info(f"Generated {len(image_pairs)} replaced images for background replacement.")
        else:
            logging.warning("No valid object detected for background replacement.")

    # 选择得分最高的图像并保存
    best_image = select_best_image_pair(image_pairs, ori_sentence,mod_sentence)
    if best_image:
        best_image_path = out_dir / "best_image.png"
        # save_array_to_img(best_image[1], best_image_path)
        best_image.save(best_image_path)
        logging.info(f"Best image saved to: {best_image_path}")
    else:
        logging.warning("No valid image pair found, skipping saving.")

if __name__ == "__main__":
    # # 手动输入参数，方便在 VSCode 中运行
    # input_img = r"D:\研究生阶段\研0\VSCode_workspace\MORE\MRE\data\img_org\train\twitter_stream_2018_07_23_3_0_2_21.jpg"
    # coords_type = 'key_in'  # "click" 或 "key_in"
    # point_coords = [400, 149]  # 选择的坐标点
    # point_labels = [1]  # 标签 1 或 0
    # text_prompt = "replace man into woman"  # 生成文本提示
    # dilate_kernel_size = 15  # 可选的膨胀大小
    # output_dir = './results'
    # sam_model_type = 'vit_h'
    # sam_ckpt = './pretrained_models/sam_vit_h_4b8939.pth'
    # seed = None  # 可选的随机种子

    # # 调用函数处理图像
    # process_image_with_sam(
    #     input_img=input_img,
    #     coords_type=coords_type,
    #     point_coords=point_coords,
    #     point_labels=point_labels,
    #     text_prompt=text_prompt,
    #     dilate_kernel_size=dilate_kernel_size,
    #     output_dir=output_dir,
    #     sam_model_type=sam_model_type,
    #     sam_ckpt=sam_ckpt,
    #     seed=seed
    # )

    # process_image_with_replacement(
    #     input_img=input_img,
    #     coords_type=coords_type,
    #     point_coords=point_coords,
    #     point_labels=point_labels,
    #     text_prompt=text_prompt,
    #     dilate_kernel_size=dilate_kernel_size,
    #     output_dir=output_dir,
    #     sam_model_type=sam_model_type,
    #     sam_ckpt=sam_ckpt,
    #     seed=seed
    # )
    # img = Image.open(input_img)
    # res = person_detect(img)
    # print(res)

    img_folder = r"D:\研究生阶段\研0\VSCode_workspace\MORE\MRE\data\img_org\train"
    caption_file = r"D:\研究生阶段\研0\VSCode_workspace\MORE\MRE\data\txt_with_caption\ours_train_with_caption.txt"
    modified_caption_file = r"D:\研究生阶段\研0\VSCode_workspace\MORE\MRE\data\txt_cap_modified\ours_train_with_caption.txt"
    output_dir = './results'
    
    dilate_kernel_size = 15
    sam_model_type = 'vit_h'
    sam_ckpt = './pretrained_models/sam_vit_h_4b8939.pth'
    seed = None
    
    # GPT client 初始化（假设已定义）
    client = OpenAI() # 这里替换为实际的GPT客户端

    # 批量处理图像
    pipeline_process_images_from_folder(
        img_folder=img_folder,
        caption_file=caption_file,
        cap_modified_file=modified_caption_file,
        output_dir=output_dir,
        dilate_kernel_size=dilate_kernel_size,
        sam_model_type=sam_model_type,
        sam_ckpt=sam_ckpt,
        seed=seed,
        client=client
    )
    logging.info("Batch image processing complete.")

    # process_single_image(
    #     input_img = r"D:\研究生阶段\研0\VSCode_workspace\MORE\MRE\data\img_org\train\twitter_stream_2018_10_10_21_0_2_202.jpg",
    #     text_prompt = " Two men standing next to each other in front of a  house",
    #     original_prompt=" Two men standing next to each other in front of a  hall",
    #     output_dir='./results',
    #     sam_model_type = 'vit_h',
    #     sam_ckpt= './pretrained_models/sam_vit_h_4b8939.pth',
    #     seed = None
    # )

