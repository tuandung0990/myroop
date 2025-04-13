import os
import cv2
import torch
from roop import face_swapper
from roop import face_helper
from roop import video_helper
from roop import image_helper
from roop import logger
from google.colab import files

def main():
    # Tải file từ Google Colab
    uploaded = files.upload()

    # Lấy các file đã tải lên
    source_file = list(uploaded.keys())[0]  # Giả sử người dùng tải lên 1 file nguồn
    target_file = list(uploaded.keys())[1]  # Giả sử người dùng tải lên 1 file mục tiêu

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load source image
    source_img = image_helper.load_image(source_file)
    if source_img is None:
        logger.error(f'Failed to load source image: {source_file}')
        return

    # Initialize face helper and swapper
    face_helper.init_models(device)
    face_swapper.init_models(device)

    # Kiểm tra xem target là ảnh hay video
    if video_helper.is_video_file(target_file):
        # Xử lý video
        video_helper.process_video(
            source_img=source_img,
            target_path=target_file,
            output_path='output.mp4',
            device=device,
            video_encoder='libx264'
        )
    else:
        # Xử lý ảnh
        target_img = image_helper.load_image(target_file)
        if target_img is None:
            logger.error(f'Failed to load target image: {target_file}')
            return

        result_img = face_swapper.swap_faces(source_img, target_img, device)
        if result_img is not None:
            cv2.imwrite('output.png', result_img)
            logger.info(f'Saved output image to output.png')
        else:
            logger.error('Face swap failed.')

if __name__ == '__main__':
    main()
