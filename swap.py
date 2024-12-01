import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
# 加载 YOLOv5 模型
model_yolo = YOLO('yolov5lu.pt')
sam_checkpoint = r"D:\Google_Browser_download\sam_vit_h_4b8939.pth"  # 替换为您的权重文件路径
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

def extract_and_swap_objects(model,image_path,save_path):
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
    results = model(image)

    # 加载 SAM 模型
    predictor.set_image(image)

    # 提取检测结果，按标签分组
    objects = {}
    for r in results:
        for box in r.boxes:
            label = int(box.cls[0])  # 类别
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
        if len(objs) < 2:
            continue  # 如果目标数少于 2，跳过此标签

        # 选择两个目标
        obj1, obj2 = objs[:2]

        # 使用 SAM 生成掩码
        masks = []
        for obj in [obj1, obj2]:
            input_points = np.array([obj["center"]])
            input_labels = np.array([1])
            mask, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)
            masks.append(mask[0])

        # 裁剪目标区域
        obj_images = []
        for i, obj in enumerate([obj1, obj2]):
            x1, y1, x2, y2 = obj["bbox"]
            cropped_image = image[y1:y2, x1:x2]
            cropped_mask = masks[i][y1:y2, x1:x2]
            obj_images.append((cropped_image, cropped_mask, obj["bbox"]))

        # 获取背景图像（去除目标区域）
        bg_mask = 1 - (masks[0] + masks[1])
        background = cv2.bitwise_and(image, image, mask=bg_mask.astype(np.uint8))

        # 调用 AnyDoor 交换位置
        # gen_image1 = inference_single_image(obj_images[0][0], obj_images[0][1], background.copy(), masks[1])
        # gen_image2 = inference_single_image(obj_images[1][0], obj_images[1][1], background.copy(), masks[0])

        # 合成最终图像
        # combined = np.where(masks[0][:, :, None], gen_image2, background)
        # combined = np.where(masks[1][:, :, None], gen_image1, combined)

        # 保存结果
        # result_image = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(save_path, result_image)
        print(f"Saved swapped image to {save_path}")
        return  # 只处理第一组，退出函数

    print("No matching objects found with at least two instances having the same label.")

image_path = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\img_org\total\0ce97a65-feb3-52ce-b4e1-dac18cb90a9f.jpg"
save_path = "./output"

extract_and_swap_objects(model_yolo,image_path,save_path)