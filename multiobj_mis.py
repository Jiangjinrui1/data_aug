import cv2
import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append('.')
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import fill_img_with_sd
from utilss import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
from ultralytics import YOLO

from diffusers import StableDiffusionInpaintPipeline

device = "cuda"
model_yolo = YOLO("/root/autodl-tmp/yolov5lu.pt")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
    ).to(device)
label_mapping = model_yolo.names



def setup_args():
    class Args:
        # input_img: str = "./example/replace-anything/dog.png"
        input_img: str = "/autodl-fs/data/data/data/MORE/img_org/total/bbde4e6c-9cad-50d8-b860-12cb800357f7.jpg"
        coords_type: str = "key_in"
        point_coords: List[float] = [750, 500]
        point_labels: List[int] = [1]
        text_prompt: str = "a man with pink clothes"
        dilate_kernel_size: int = 15
        output_dir: str = "./results"
        sam_model_type: str = "vit_h"
        sam_ckpt: str = "/root/autodl-tmp/sam_vit_h_4b8939.pth"
        seed: int = None
        deterministic: bool = False
    return Args()

if __name__ == "__main__":
    """Example usage:
    python fill_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --coords_type key_in \
        --point_coords 750 500 \
        --point_labels 1 \
        --text_prompt "a teddy bear on a bench" \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth 
    """
    args = setup_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    img = load_img_to_array(args.input_img)

    results = model_yolo(img)
    latest_coords = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            cls = int(box.cls[0])
            center = ((x1+x2)/2, (y1+y2)/2)
            if label_mapping[cls] == 'tie':
                latest_coords = center
                break
        

    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        args.point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # fill the masked image
    for idx, mask in enumerate(masks):
        if args.seed is not None:
            torch.manual_seed(args.seed)
        mask_p = out_dir / f"mask_{idx}.png"
        img_filled_p = out_dir / f"filled_with_{Path(mask_p).name}"
        img_filled = fill_img_with_sd(
            img, mask, args.text_prompt, device=device,pipe=pipe)
        save_array_to_img(img_filled, img_filled_p)