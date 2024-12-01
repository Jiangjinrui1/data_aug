import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['http__proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

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

from openai import OpenAI

client = OpenAI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置日志
logging.basicConfig(
    filename='process_log_img.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 加载CLIP模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
    ).to(device)
lama_config = "./lama/configs/prediction/default.yaml"
lama_ckpt = "./pretrained_models/big-lama"

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

def generate_prompt_for_caption(client, original_caption, modified_caption):
    """
    根据 caption 的变化情况生成 prompt：
    - 若无变化，生成与原始 caption 有微小差异的描述。
    - 若有变化，生成反映修改内容的 prompt。
    """
    # 判断 caption 是否发生变化
    if original_caption == modified_caption:
        # Caption 无变化，生成微小差异的描述
        prompt_request = f"""
        You are an assistant generating slight variations for a description.
        Based on the sentence: "{original_caption}", create several options with minor differences.
        Introduce small changes such as slight clothing adjustments, different expressions, or subtle background details. 
        Avoid changing the main subject.
        """
    else:
        # Caption 有变化，生成包含变化内容的 prompt
        prompt_request = f"""
        You are an assistant generating descriptions with minor modifications.
        Original sentence: "{original_caption}"
        Modified sentence: "{modified_caption}"
        
        Create several prompts that reflect the changes in the modified sentence. 
        Highlight minor adjustments, such as changes in appearance, clothing, or objects held. Keep it concise.
        """

    # 请求生成 prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_request}],
        max_tokens=15
    )
    prompts = response.choices[0].message.content.split("\n")
    return [p.strip() for p in prompts if p.strip()]

def select_best_prompt(original_sentence, candidates, clip_model, clip_processor):
    """
    从多个候选 prompt 中选择与原始句子相似度次高的 prompt（适用于有变化的情况）。
    若仅有一个候选项或无变化，则返回相似度最高的 prompt。
    """
    inputs = clip_processor(text=[original_sentence] + candidates, return_tensors="pt", padding=True)
    outputs = clip_model.get_text_features(**inputs)

    # 获取原始句子和候选 prompt 的嵌入
    original_embedding = outputs[0]
    candidate_embeddings = outputs[1:]

    # 计算相似度
    similarities = torch.nn.functional.cosine_similarity(original_embedding.unsqueeze(0), candidate_embeddings)

    # 若候选项数量少于 2，直接返回最高相似度的候选项
    if len(similarities) < 2:
        return candidates[similarities.argmax().item()]

    # 选择次高相似度的 prompt
    _, top_indices = similarities.topk(2, largest=True)
    second_best_index = top_indices[1].item()
    return candidates[second_best_index]

# 主函数：生成并选择最佳 prompt
def generate_entity_replacement_prompt(client, original_caption, modified_caption, clip_model, clip_processor):
    # 基于 caption 变化情况生成候选 prompt
    candidate_prompts = generate_prompt_for_caption(client, original_caption, modified_caption)

    # 判断是否需要选择次高相似度的 prompt
    if original_caption == modified_caption:
        best_prompt = candidate_prompts[0]  # 无变化时，直接返回生成的微小差异 prompt
    else:
        # 有变化时选择次高相似度的 prompt
        best_prompt = select_best_prompt(original_caption, candidate_prompts, clip_model, clip_processor)

    print(f"Final selected prompt: {best_prompt}")
    return best_prompt
from PIL import Image, ImageDraw

# 预览选中的对象
def preview_object_box(img_path, obj_box):
    img = Image.open(img_path)
    img_width, img_height = img.size

    # 将相对坐标转换为像素坐标
    x_min = int(obj_box[0] * img_width)
    y_min = int(obj_box[1] * img_height)
    x_max = int(obj_box[2] * img_width)
    y_max = int(obj_box[3] * img_height)

    # 在图像上绘制矩形框
    draw = ImageDraw.Draw(img)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # 显示图像
    img.show()
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
        # 若输出文件夹已经包含图像，则跳过
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
        operation = "fill"  # 默认操作为填充

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
            
            # 计算边框面积
            # 读取中心点和宽高
            center_x, center_y, box_width, box_height = obj_box

            x_min = center_x - box_width / 2
            y_min = center_y - box_height / 2
            x_max = center_x + box_width / 2
            y_max = center_y + box_height / 2

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(x_max,1)
            y_max = min(y_max,1)

            obj_box = (x_min, y_min, x_max, y_max)
            area = (x_max - x_min) * (y_max - y_min)

            # 更新面积最大对象
            if area > max_area:
                max_area = area
                largest_obj_id = obj_id
                largest_obj_box = obj_box
                largest_original_caption = original_caption
                largest_modified_caption = modified_caption
            # 若找到面积最大的对象，调用处理函数
        if largest_obj_id:
            preview_object_box(img_path, largest_obj_box)
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

def process_single_object(
    input_img: str,
    text_prompt: str,
    original_prompt: str,
    output_dir: str,
    obj_box: tuple,
    sam_model_type: str = 'vit_h',
    sam_ckpt: str = 'sam_vit_h_4b8939.pth',
    operation: str = 'fill',
    seed: int = None,
    dilate_kernel_size = 15
):
    """处理单个 OBJ 实体的图像，并生成带遮罩的图像"""
    logging.info(f"Processing image {input_img} for object with bounding box {obj_box}")
    modified_caption = text_prompt

    img = load_img_to_array(input_img)
    img_height, img_width = img.shape[:2]
    
    # 根据原始图像尺寸，将相对坐标转换为绝对坐标
    abs_box = (
        int(obj_box[0] * img_width),
        int(obj_box[1] * img_height),
        int(obj_box[2] * img_width),
        int(obj_box[3] * img_height)
    )
    
    # 获取 OBJ 中心坐标作为 SAM 的点击点
    center_x = (abs_box[0] + abs_box[2]) // 2
    center_y = (abs_box[1] + abs_box[3]) // 2
    # center_x, center_y, box_width, box_height = obj_box
    # 将相对坐标转换为绝对坐标
    # center_x = int(center_x * img_width)
    # center_y = int(center_y * img_height)
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
    if operation == "fill":
        text_prompt = generate_entity_replacement_prompt(client,original_prompt, modified_caption,clip_model,clip_processor)
        logging.info(f"Original prompt: {original_prompt}")
        logging.info(f"Modified prompt: {modified_caption}")
        logging.info(f"Generated text prompt: {text_prompt}")
        # 使用 Stable Diffusion 生成新的图像
        for idx, mask in enumerate(masks):
            if seed is not None:
                torch.manual_seed(seed)
            img_filled = fill_img_with_sd(img, mask, text_prompt, device='cuda' if torch.cuda.is_available() else 'cpu',pipe = pipe)
            
            if img_filled is None:
                logging.warning(f"Failed to generate filled image for {input_img} with object {obj_box}")
                continue
            candidate_pairs.append((img,img_filled))
        best_pair = select_best_image_pair(candidate_pairs,original_prompt,modified_caption)
            # 计算相似度和CLIPdir
            
        save_path = out_dir / f"{img_stem}.png"
        best_pair.save(save_path)
    elif operation == "remove":
        if seed is not None:
            torch.manual_seed(seed)
        img_removed = inpaint_img_with_lama(img, masks[2], config_p = lama_config, ckpt_p= lama_ckpt, device=device)
        # candidate_pairs.append((img,img_removed))
        # best_pair = select_best_image_pair(candidate_pairs,original_prompt,modified_caption)
        save_path = out_dir / f"{img_stem}.png"
        save_array_to_img(img_removed, save_path)              
 
    # save_array_to_img(best_pair, save_path)

    logging.info(f"Saved filled image to {save_path}")

# 示例使用方法
if __name__ == "__main__":
    img_folder = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\img_org\train"
    txt_file = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\txt/train.txt"
    caption_file = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\caption_dict.json"
    cap_modified_file = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\caption_modified.json"  # 修改后的 caption 文件
    output_dir = './results_more/train'
    
    dilate_kernel_size = 15
    sam_model_type = 'vit_h'
    sam_ckpt = './pretrained_models/sam_vit_h_4b8939.pth'
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
