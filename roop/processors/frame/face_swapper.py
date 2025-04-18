from typing import Any, List
import cv2
import insightface
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
NAME = 'ROOP.FACE-SWAPPER'

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('models')  # Đảm bảo đúng đường dẫn
    conditional_download(download_directory_path, ['https://huggingface.co/tuandung/inswapper/resolve/main/inswapper_128.onnx'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    if FACE_SWAPPER is None:
        model_path = resolve_relative_path('models/inswapper_128.onnx')  # Đảm bảo đúng đường dẫn trong Colab
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    with ThreadPoolExecutor() as executor:
        futures = []
        for temp_frame_path in temp_frame_paths:
            future = executor.submit(process_frame, source_face, cv2.imread(temp_frame_path))
            futures.append(future)
        for future in futures:
            result = future.result()
            cv2.imwrite(temp_frame_path, result)
            if progress:
                progress.update(1)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    result = process_frame(source_face, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    from roop.processors.frame.core import process_video
    process_video(source_path, temp_frame_paths, process_frames)
