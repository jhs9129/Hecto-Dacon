import os
import time
import torch
import logging
import argparse

from PIL import Image
from rich.logging import RichHandler
from torchvision.transforms import v2
from torchvision.transforms import ToPILImage
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='') # 이미지 데이터 파일 경로
parser.add_argument('--save_dir', default='') # 이미지 증강 데이터 저장 파일 이름
parser.add_argument('--num_workers', type=int, default=2)

args = parser.parse_args()
logger.info(args)


augmentation_transforms = {
    '원본': v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),

    '수평반전': v2.Compose([
        v2.RandomHorizontalFlip(p=1),
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),

    # '랜덤 회전': v2.Compose([
    #     v2.RandomRotation(degrees=30),
    #     v2.Resize((224, 224)),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True)
    # ]),

    '색상,밝기,채도1': v2.Compose([
        v2.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2),
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),

    '색상,밝기,채도2': v2.Compose([
        v2.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2),
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),

    '색상,밝기,채도3': v2.Compose([
        v2.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2),
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),

    '랜덤 크롭': v2.Compose([
        v2.RandomResizedCrop((224, 224), scale=(0.1, 0.5), ratio=(1.2, 1.5)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),

    # '랜덤 크롭2': v2.Compose([
    #     v2.RandomResizedCrop((224, 224), scale=(0.1, 0.5), ratio=(1.2, 1.5)),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True)
    # ]),

    # '랜덤 크롭3': v2.Compose([
    #     v2.RandomResizedCrop((224, 224), scale=(0.1, 0.5), ratio=(1.2, 1.5)),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True)
    # ]),

    # '아핀 변환': v2.Compose([
    #     v2.RandomAffine(
    #         degrees=20,
    #         translate=(0.05, 0.1),
    #         scale=(0.9, 1.1)),
    #     v2.Resize((224, 224)),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True)
    # ]),

    '가우시안 블러': v2.Compose([
        v2.GaussianBlur(kernel_size=5, sigma=100),
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),

    # '가우시안 블러2': v2.Compose([
    #     v2.GaussianBlur(kernel_size=5, sigma=100),
    #     v2.Resize((224, 224)),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True)
    # ]),

    # '가우시안 블러3': v2.Compose([
    #     v2.GaussianBlur(kernel_size=5, sigma=100),
    #     v2.Resize((224, 224)),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True)
    # ]),

    '회색조': v2.Compose([
        v2.RandomGrayscale(p=1),
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]),

    # '랜덤 원근변환1': v2.Compose([
    #     v2.RandomPerspective(distortion_scale=0.1, p=1),
    #     v2.Resize((224, 224)),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True)
    # ]),

    # '랜덤 원근변환2': v2.Compose([
    #     v2.RandomPerspective(distortion_scale=0.3, p=1),
    #     v2.Resize((224, 224)),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True)
    # ])
}
# 이미지 4등분(좌우 / 상하)
def split_center_lines(image: Image.Image):
    width, height = image.size
    # 세로 중앙선 분할
    left_img = image.crop((0, 0, width // 2, height))
    right_img = image.crop((width // 2, 0, width, height))
    # 가로 중앙선 분할
    top_img = image.crop((0, 0, width, height // 2))
    bottom_img = image.crop((0, height // 2, width, height))
    return [left_img, right_img, top_img, bottom_img]

# 단일 이미지에 대해 증강을 적용하고 저장
def process_and_save(img_path, label_name, save_dir, augmentation_transforms):
    try:
        image = Image.open(img_path).convert('RGB')
        to_pil = ToPILImage()
        results = 0
        for aug_name, transform in augmentation_transforms.items():
            tensor_img = transform(image)
            pil_img = to_pil(tensor_img)
            save_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{aug_name}.jpg"
            save_path = os.path.join(save_dir, label_name, save_name)
            pil_img.save(save_path)
            results += 1
        split_data = split_center_lines(image)
        for split_name, split_img in zip(["left", "right", "top", "bottom"], split_data):
            save_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{split_name}.jpg"
            save_path = os.path.join(save_dir, label_name, save_name)
            split_img.save(save_path)
            results += 1


    except Exception as e:
        logger.info(f"Error processing {img_path}: {e}")
        return 0

# 전체 증강
def main(image_dir, save_dir, augmentation_transforms, num_workers=2):
    start = time.time()
    label_count = 0

    # 병렬 처리
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        
        for label_name in os.listdir(image_dir):
            
            if label_name not in os.listdir(save_dir):
                os.makedirs(os.path.join(save_dir, label_name), exist_ok=True)

            label_count += 1
            data_path = os.path.join(image_dir, label_name)
            if len(os.listdir(os.path.join(save_dir, label_name))) == 0:
                image_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.lower().endswith('jpg')]
                
                futures = []
                for img_path in image_files:

                    future = executor.submit(process_and_save, img_path, label_name, save_dir, augmentation_transforms)
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result()  

                logger.info(f"{label_name} 데이터 생성 완료하였습니다.")
            
            logger.info(f"{len(os.listdir(image_dir))}개의 label 중 {label_count}건 증강 완료")

    end = time.time()
    logger.info(f"data augmentation - {end - start: .2f}초 소요되었습니다.")

if __name__ == "__main__":
    main(args.image_dir, args.save_dir, augmentation_transforms, args.num_workers)