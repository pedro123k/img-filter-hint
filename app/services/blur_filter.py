import numpy as np
import cv2 


def remove_blur(img_raw: bytes) -> bytes:
    ndarray_buff = np.frombuffer(img_raw, dtype=np.uint8)
    img = cv2.imdecode(ndarray_buff, cv2.IMREAD_COLOR)
    
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(img, 1.3, blur, -0.5, 0)

    success, encoded = cv2.imencode(".jpg", sharpened)
    if not success:
        raise RuntimeError("Encoding Error!")

    return encoded.tobytes()