import cv2
import numpy as np

def lighten(img_raw: bytes) -> bytes:

    ndarray_buff = np.frombuffer(img_raw, dtype=np.uint8)
    img = cv2.imdecode(ndarray_buff, cv2.IMREAD_COLOR)

    gamma = 0.5
    inv_gamma = 1.0 / gamma

    t = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype(np.uint8)

    img_gamma = cv2.LUT(img, t)

    img_lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    L_clahe = clahe.apply(L)

    img_lab_clahe = cv2.merge((L_clahe, A, B))
    img_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2BGR)

    success, encoded = cv2.imencode(".jpg", img_clahe)
    if not success:
        raise RuntimeError("Encoding Error!")

    return encoded.tobytes()
