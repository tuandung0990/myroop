from typing import Any, List
import cv2
import gfpgan
import roop.globals
from roop.core import update_status
from roop.face_analyser import get_one_face
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

FACE_ENHANCER = None
NAME = 'ROOP.FACE-ENHANCER'

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('models')  # Đảm bảo đúng đường dẫn
    conditional_download(download_directory_path, ['https://huggingface.co/tuandung/inswapper/resolve/main/GFPGANv1.4.pth'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    if FACE_ENHANCER is None:
        model_path = resolve_relative_path('models/GFPGANv1.4.pth')  # Đảm bảo đúng đường dẫn trong Colab
        FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1)  # type: ignore[attr-defined]
    return FACE_ENHANCER

def enhance_face(temp_frame: Frame) -> Frame:
    _, _, temp_frame = get_face_enhancer().enhance(temp_frame, paste_back=True)
    return temp_frame

def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    with ThreadPoolExecutor() as executor:
        futures = []
        for temp_frame_path in temp_frame_paths:
            future = executor.submit(process_frame, None, cv2.imread(temp_frame_path))
            futures.append(future)
        for future in futures:
            result = future.result()
            cv2.imwrite(temp_frame_path, result)
            if progress:
                progress.update(1)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    from roop.processors.frame.core import process_video
    process_video(source_path, temp_frame_paths, process_frames)
