from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import shutil
from pathlib import Path
from app.predictor import classify_image
from app.evaluator import evaluate_model_on_folder  # เพิ่มบรรทัดนี้ด้านบน
import os  # 👈 เพิ่มบรรทัดนี้

app = FastAPI()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

FEEDBACK_DIR = Path("feedback_data")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    img_np = np.array(img)

    result = classify_image(img_np)

    fake_prob = round(result["probabilities"].get("fake", 0), 2)
    real_prob = round(result["probabilities"].get("real", 0), 2)

    temp_path = Path("static/temp") / image.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(temp_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result["prediction"].capitalize(),
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "filename": image.filename,
        "image_path": f"/static/temp/{image.filename}"
    })

@app.post("/submit_feedback")
async def submit_feedback(filename: str = Form(...), correct_label: str = Form(...)):
    try:
        temp_path = Path("static/temp") / filename
        if not temp_path.exists():
            return JSONResponse(content={"status": "error", "message": "Image not found"}, status_code=400)

        dest_dir = FEEDBACK_DIR / correct_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Move the image file
        shutil.move(str(temp_path), dest_dir / filename)

        # Optional: Save label info
        

        return JSONResponse(content={"status": "success"})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.get("/metrics")
async def get_model_metrics():
    real_test_path = "test_dataset/Real"
    fake_test_path = "test_dataset/Fake"

    if not os.path.exists(real_test_path) or not os.path.exists(fake_test_path):
        return JSONResponse(status_code=400, content={"error": "ไม่พบโฟลเดอร์ test/Real หรือ test/Fake"})

    report, cm_img = evaluate_model_on_folder(real_test_path, fake_test_path)

    return JSONResponse(content={
        "accuracy": round(report["accuracy"], 4),
        "real": {
            "precision": round(report["Real"]["precision"], 4),
            "recall": round(report["Real"]["recall"], 4),
            "f1_score": round(report["Real"]["f1-score"], 4)
        },
        "fake": {
            "precision": round(report["Fake"]["precision"], 4),
            "recall": round(report["Fake"]["recall"], 4),
            "f1_score": round(report["Fake"]["f1-score"], 4)
        },
        "confusion_matrix_base64": cm_img
    })
