from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.xception import preprocess_input
import numpy as np
import shutil
import os
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# โหลดโมเดลที่เทรนไว้
model = load_model("model/deepfake_xception.h5")

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    return label, float(prediction)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}_{file.filename}"
    upload_path = f"static/{filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label, confidence = predict_image(upload_path)
    os.remove(upload_path)

    return JSONResponse(content={
        "label": label,
        "confidence": round(confidence, 4)
    })
