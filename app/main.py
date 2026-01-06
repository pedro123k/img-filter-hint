import fastapi
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from . import services
import time 
from uuid import uuid4
import numpy as np
import cv2
import torch
import base64
import json

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent /  "model/model.pth"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
img_cache = services.ImgCache()
cnnModel = services.CNNCustomModel()
cnnModel.load_state_dict(torch.load(MODEL_DIR))
cnnModel.to('cuda' if torch.cuda.is_available() else 'cpu')

templates = Jinja2Templates(BASE_DIR / "templates")

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/review", response_class=HTMLResponse)
def review(request: Request):
    return templates.TemplateResponse(
        "review.html",
        {"request": request}
    )

@app.get("/model_info", response_class=HTMLResponse)
def model_info(request: Request):
    summary = None
    cm = None

    with open(BASE_DIR.parent / "results/summary.json") as f:
        summary = json.load(f)
    
    with open(BASE_DIR.parent / "results/confusion_matrix.json") as f:
        cm = json.load(f)

    cm_norm = (np.array(cm) / np.sum(cm, axis=1, keepdims=True)).tolist()

    return templates.TemplateResponse(
        "model_info.html",
        {"request": request, "cm": cm_norm, "summary":summary}
    )

@app.get("/api/evaluation/{uuid}")
def evaluation(uuid: str):
    temp = img_cache[uuid]
    pred = services.predict(cnnModel, temp["content"])

    LABELS = ["OK", "NOISY", "BLURRED", "LOW-LIGHTENED"]
    greater_idx = pred.index(max(pred))

    if temp is not None:
        return {
            "ok": True,
            "content": {
                "score": pred[0],
                "label": LABELS[greater_idx]
            }
        }
    else:
        raise HTTPException(status_code=404, detail="Image Not Found")

@app.get("/api/filter_noise/{uuid}")
def filter_noise(uuid: str):
    temp = img_cache[uuid]

    if temp is None:
        raise HTTPException(status_code=404, detail="Image Not Found")
    
    filtered = services.remove_noise(temp["content"])
    pred = services.predict(cnnModel, filtered)
    LABELS = ["OK", "NOISY", "BLURRED", "LOW-LIGHTENED"]
    greater_idx = pred.index(max(pred))

    return {
            "ok": True,
            "content": {
                "img_result": base64.b64encode(filtered).decode("utf-8"),
                "score": pred[0],
                "label": LABELS[greater_idx]
            }
        }

@app.get("/api/filter_blur/{uuid}")
def filter_blur(uuid: str):
    temp = img_cache[uuid]

    if temp is None:
        raise HTTPException(status_code=404, detail="Image Not Found")
    
    filtered = services.remove_blur(temp["content"])
    pred = services.predict(cnnModel, filtered)
    LABELS = ["OK", "NOISY", "BLURRED", "LOW-LIGHTENED"]
    greater_idx = pred.index(max(pred))

    return {
            "ok": True,
            "content": {
                "img_result": base64.b64encode(filtered).decode("utf-8"),
                "score": pred[0],
                "label": LABELS[greater_idx]
            }
        }

@app.get("/api/filter_light/{uuid}")
def filter_light(uuid: str):
    temp = img_cache[uuid]

    if temp is None:
        raise HTTPException(status_code=404, detail="Image Not Found")
    
    filtered = services.lighten(temp["content"])
    pred = services.predict(cnnModel, filtered)
    LABELS = ["OK", "NOISY", "BLURRED", "LOW-LIGHTENED"]
    greater_idx = pred.index(max(pred))

    return {
            "ok": True,
            "content": {
                "img_result": base64.b64encode(filtered).decode("utf-8"),
                "score": pred[0],
                "label": LABELS[greater_idx]
            }
        }
        

@app.get("/api/imgs_cached/{uuid}")
def img_stored(uuid: str):
    temp = img_cache[uuid]
    if temp is not None:
        return Response(content=temp["content"], media_type=temp["content-type"])
    else:
        raise HTTPException(status_code=404, detail="Image Not Found")

@app.post("/upload_img")
async def upload_img(file: UploadFile = File(...)):
    data = await file.read()
    npbuf = np.frombuffer(data, np.uint8)

    img = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Image encoding error!")
    
    new_uuid = str(uuid4())
    
    img_cache.add_img(
        uuid = new_uuid,
        img = data,
        content_type = file.content_type,
        date = time.time()
    )

    return {
        "ok": True,
        "uuid": new_uuid
    }