import os
import json
import ast
import cv2
import numpy as np

# from remove_and_swap import remove_and_swap
# from resize import resize_img
# from anydoor_swap import extract_and_swap_objects


from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import openai

model_yolo = YOLO("/root/autodl-tmp/yolov5lu.pt")
vectorizer = TfidfVectorizer()
# openai.api_key = 'your-api-key'



url_text = r"/autodl-fs/data/data/data/MORE/txt/train.txt"
url_json = r"/autodl-fs/data/data/data/MORE/caption_dict.json"


# load_text
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        txt_data = f.readlines()
        txt_data = [ast.literal_eval(line) for line in txt_data] 
    return txt_data
def load_caption_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def preprocess_data(text_data, caption_data):
    # 将text_data转换为字典，以img_id为键
    text_dict = {entry['img_id']: entry['text'] for entry in text_data}
    caption_dict = caption_data
    
    return text_dict, caption_dict

def get_text_and_caption_by_img_id(img_id, text_dict, caption_dict):
    # 通过img_id获取对应的文本内容和caption内容
    text_entries = text_dict.get(img_id, [])
    caption_entries = caption_dict.get(img_id, {})
        # 只提取caption的值
    # caption_values = list(caption_entries.values())
    caption_values = ' '.join(caption_entries.values())  # 使用空格连接所有的caption
    
    return text_entries, caption_values

text_data = load_text(url_text)
caption_data = load_caption_json(url_json)
text_dict, caption_dict = preprocess_data(text_data, caption_data)

# size_detect
def get_openai_similarities(text, yolov5_labels):
    """
    使用OpenAI GPT模型来生成与YOLOv5标签的相似度
    """
    # 使用OpenAI API进行匹配
    prompt = f"Given the following description of an object: '{text}', which of these labels from the list below best describes it?\nLabels: {', '.join(yolov5_labels)}\nAnswer:"
    client = client_size
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 或使用 "gpt-3.5-turbo" 等
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )
    
    return response.choices[0].text.strip()

def match_text_with_yolov5_using_openai(text_entries, caption_values, results):
    """
    匹配文本中的物体与YOLOv5检测结果，并使用OpenAI进行智能匹配
    """
    matched_labels = []

    # 获取YOLOv5的检测结果中的标签名称
    yolov5_labels = [box.cls[0].cpu().numpy() for result in results for box in result.boxes]
    label_mapping = model_yolo.names  # 获取YOLOv5的标签映射
    yolov5_label_names = [label_mapping[label] for label in yolov5_labels]

    # 排除掉 "person" 类别
    yolov5_label_names = [label for label in yolov5_label_names if label != 'person']
    
    # 使用OpenAI API来智能匹配标签
    for text in text_entries + caption_values:
        # 使用OpenAI API进行智能匹配
        matched_label = get_openai_similarities(text, yolov5_label_names)
        matched_labels.append(matched_label)

    return matched_labels

def detect_objects_and_match_text_with_openai(img, text_entries, caption_values):
    """
    通过YOLOv5检测图片中的物体，并根据文本匹配相关物体，使用OpenAI增强智能匹配
    """
    # 获取YOLOv5的检测结果
    results = model_yolo(img)

    # 识别文本中的物体与YOLOv5的检测结果匹配
    matched_labels = match_text_with_yolov5_using_openai(text_entries, caption_values, results)

    return matched_labels
client_size = OpenAI()



if __name__ == "__main__":

    a,b = get_text_and_caption_by_img_id("3cc2e961-69a3-5ca8-92b9-9eb057e139df.jpg", text_dict, caption_dict)
    img = "/autodl-fs/data/data/data/MORE/img_org/total/3cc2e961-69a3-5ca8-92b9-9eb057e139df.jpg"
    matched_labels = detect_objects_and_match_text_with_openai(img, a, b)
    print(matched_labels)
    # print(a,b)





