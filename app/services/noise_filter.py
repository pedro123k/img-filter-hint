import numpy as np
import cv2

def remove_noise(img_raw: bytes) -> bytes:
    ndarray_buff = np.frombuffer(img_raw, dtype=np.uint8)
    img = cv2.imdecode(ndarray_buff, cv2.IMREAD_COLOR)

    img = cv2.medianBlur(img, 3)
    img = cv2.bilateralFilter(img, 9, 55, 45)

    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Encoding Error!")

    return encoded.tobytes()