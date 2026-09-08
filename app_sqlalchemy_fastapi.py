from fastapi import FastAPI, UploadFile, File
import os
import shutil

from database import engine, Base
from database.models import Image

app = FastAPI()

# Create database tables
Base.metadata.create_all(bind=engine)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "FastAPI + SQLite is connected"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Image uploaded successfully",
        "filename": file.filename,
        "path": file_path
    }
