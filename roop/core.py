#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
import torch
import onnxruntime
import tensorflow
import platform
import signal
import warnings
from typing import List

# Check GPU availability in Colab
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 if available
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU if no GPU available

# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

import roop.globals
import roop.metadata
from roop.predicter import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

# Function to parse arguments for Colab
def parse_args() -> None:
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select a source image', dest='source_path')
    program.add_argument('-t', '--target', help='select a target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    # (Other arguments remain the same...)

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)

    # Handle deprecated arguments
    if args.source_path_deprecated:
        print('Argument -f and --face are deprecated. Use -s and --source instead.')
        roop.globals.source_path = args.source_path_deprecated
    if args.cpu_cores_deprecated:
        print('Argument --cpu-cores is deprecated. Use --execution-threads instead.')
        roop.globals.execution_threads = args.cpu_cores_deprecated

# Replace UI components with print statements
def update_status(message: str) -> None:
    print(f'ROOP.CORE: {message}')

# Simplify the start function for Colab (No UI interaction)
def start() -> None:
    update_status('Starting processing...')
    # Process images or videos here
    # For image processing
    if has_image_extension(roop.globals.target_path):
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        update_status('Image processing complete!')
    # For video processing
    else:
        update_status('Processing video...')
        create_temp(roop.globals.target_path)
        extract_frames(roop.globals.target_path)
        temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Processing video frames...')
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            release_resources()
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
            restore_audio(roop.globals.target_path, roop.globals.output_path)
        clean_temp(roop.globals.target_path)
        update_status('Video processing complete!')

# Run the process on Colab
def run() -> None:
    parse_args()
    if not pre_check():
        return
    limit_resources()
    start()
