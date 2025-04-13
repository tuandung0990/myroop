rom typing import Any
import insightface
import roop.globals
from roop.typing import Frame

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
        # Kiểm tra xem có GPU không, nếu có, sử dụng GPU (Colab sẽ tự động sử dụng GPU nếu có)
        ctx_id = 0 if roop.globals.execution_providers and 'cuda' in roop.globals.execution_providers else -1
        FACE_ANALYSER.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None
