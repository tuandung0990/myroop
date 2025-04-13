import argparse
import os
import cv2
import torch
from roop import face_swapper
from roop import face_helper
from roop import video_helper
from roop import image_helper
from roop import logger

def parse_args():
    parser = argparse.ArgumentParser(description='Face Swap CLI for Colab')
    parser.add_argument('--source', type=str, required=True, help='Path to source image')
    parser.add_argument('--target', type=str, required=True, help='Path to target image or video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save output')
    parser.add_argument('--execution-provider', type=str, default='cuda', choices=['cuda', 'cpu'], help='Execution provider')
    parser.add_argument('--output-video-encoder', type=str, default='libx264', help='Video encoder for output')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set device
    device = torch.device('cuda' if args.execution_provider == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load source image
    source_img = image_helper.load_image(args.source)
    if source_img is None:
        logger.error(f'Failed to load source image: {args.source}')
        return

    # Initialize face helper and swapper
    face_helper.init_models(device)
    face_swapper.init_models(device)

    # Check if target is image or video
    if video_helper.is_video_file(args.target):
        # Process video
        video_helper.process_video(
            source_img=source_img,
            target_path=args.target,
            output_path=args.output,
            device=device,
            video_encoder=args.output_video_encoder
        )
    else:
        # Process image
        target_img = image_helper.load_image(args.target)
        if target_img is None:
            logger.error(f'Failed to load target image: {args.target}')
            return

        result_img = face_swapper.swap_faces(source_img, target_img, device)
        if result_img is not None:
            cv2.imwrite(args.output, result_img)
            logger.info(f'Saved output image to {args.output}')
        else:
            logger.error('Face swap failed.')

if __name__ == '__main__':
    main()

