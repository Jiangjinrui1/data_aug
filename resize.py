import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utilss import dilate_mask

model = YOLO('/root/autodl-tmp/yolov5lu.pt')
# model = YOLO(r'D:\VSCodeWorkSpace\Inpaint-Anything-main\yolov5lu.pt')
sam_checkpoint = "/root/autodl-tmp/sam_vit_h_4b8939.pth" 
lama_ckpt = "/root/autodl-tmp/big-lama"
device = "cuda:0"
label_mapping = model.names

def set_args(input_img = None,input_folder = None):
    class Args:
        def __init__(self):
            self.input_img = input_img
            self.input_folder = input_folder
            self.output_folder = "/root/autodl-fs/output"
            self.dilate_kernel_size = 15
            self.point_labels = [1]
            self.sam_ckpt = sam_checkpoint
            self.sam_model_type = "vit_h"
    return Args()
def get_mask(img,coords,args):
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
    rgba_image[:, :, 3] = masks[2]

    return rgba_image,masks
def resize_img(img,args):

    # Load the image
    if args.input_img is not None:
        img = cv2.imread(args.input_img)
    else:
        img = cv2.imread(img)

    results = model(img)
    for result in results:
        for box in result.boxes:
            label = box.cls.cpu().numpy()
            label_name = label_mapping[int(label)]
            if label_name == "person":
                continue
            x1,y1,x2,y2 = map(int,box.xyxy[0].cpu().numpy())
            
            scale_factor = np.random.choice([1.5,0.5])
            cropped_object = img[y1:y2, x1:x2]
            new_width = int(cropped_object.shape[1] * scale_factor)
            new_height = int(cropped_object.shape[0] * scale_factor)
            resized_object = cv2.resize(cropped_object, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            # 计算中心点cropped_object的中心点
            center_x = int((resized_object.shape[0] // 2))
            center_y = int((resized_object.shape[1]// 2))
            coords = (center_x, center_y)
            rgba_object,masks = get_mask(resized_object,coords,args)

            # 将object从原图去除
            if args.dilate_kernel_size is not None:
                masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]
            img_inpainted = inpaint_img_with_lama(
            img, masks[2], args.lama_config, args.lama_ckpt, device=device)
            # 将rgba_object粘贴到img_inpainted的指定位置
            img = img_inpainted
            bg = Image.fromarray(img,mode="RGB")
            fg = Image.fromarray(rgba_object,mode="RGBA")
            # pos = (x1, y1)
            new_pos = (x1 * scale_factor, y1 * scale_factor)
            bg.paste(fg, new_pos, fg)
            img = np.array(bg)

    return results

    
if __name__ =="__main__":
    args = set_args(input_img = r"D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\img_org\total\5a57b298-d2c4-5f34-ae93-257b6d808d24.jpg")
    result = resize_img(args.input_img,args)
        
