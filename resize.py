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
            self.lama_config = "./lama/configs/prediction/default.yaml"
            self.lama_ckpt = lama_ckpt  
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
    return masks
def get_obj(img,coords,args):
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

    return rgba_image,masks
def resize_and_mask_object(img, box, coords_bg,scale_factor, args):

    x1,y1,x2,y2 = map(int,box.xyxy[0].cpu().numpy())
    cropped_object = img[y1:y2, x1:x2]



    cropped_masks = get_mask(img, coords_bg, args)


    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    new_obj_w = int(cropped_object.shape[1] * scale_factor)
    new_obj_h = int(cropped_object.shape[0] * scale_factor)
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
            scale_factor = np.random.choice([3,0.6])
            # 计算中心点cropped_object的中心点
            center_x_bg = int((x1+x2)/2)
            center_y_bg = int((y1+y2)/2)
            coords_bg = (center_x_bg,center_y_bg)
            # rgba_object,masks_obj = get_obj(resized_object,coords,args)
            rgba_object,masks_obj = resize_and_mask_object(img,box,coords_bg,scale_factor,args)


            cv2.imwrite("./resize_tmp/mask_obj.jpg",masks_obj)
            rgba_obj = Image.fromarray(rgba_object,mode= "RGBA")
            rgba_obj.save("./resize_tmp/rgba_obj.png")
            masks = get_mask(img,coords_bg,args)
            cv2.imwrite("./resize_tmp/mask_0.jpg",masks[1])

            # 将object从原图去除
            if args.dilate_kernel_size is not None:
                masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]
            img_inpainted = inpaint_img_with_lama(
            img, masks[1], args.lama_config, args.lama_ckpt, device=device)

            img = img_inpainted
            cv2.imwrite("./resize_tmp/img_inpat.jpg",img)
            bg = Image.fromarray(img,mode="RGB")
            fg = Image.fromarray(rgba_object,mode="RGBA")
            pos = (x1, y1)
            offset = ((coords_bg[0] - pos[0]) * scale_factor,(coords_bg[1] - pos[1]) * scale_factor)
            new_pos = (int(coords_bg[0] - offset[0]),int(coords_bg[1] - offset[1]))

            bg.paste(fg, new_pos, fg)
            img = np.array(bg)
    cv2.imwrite("./resize_tmp/resized_img.jpg",img)
    return results

    
if __name__ =="__main__":
    args = set_args(input_img = r"/autodl-fs/data/data/data/MORE/img_org/total/5a57b298-d2c4-5f34-ae93-257b6d808d24.jpg")
    result = resize_img(args.input_img,args)
        
# sftp://root@connect.cqa1.seetacloud.com:29668/autodl-fs/data/data/data/MORE/img_org/total