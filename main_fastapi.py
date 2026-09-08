from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pathlib import Path
import uvicorn

from utils.path_manager import PathManager
from utils.database import FacesDatabase

app = FastAPI(title="FaceRecognition Upload API")

paths = PathManager(root_dir='.')
faces_db = FacesDatabase(paths.faces_db_file())


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), person_id: str = Form(default="unknown")):
    """Upload an image and store it in SQLite as a blob. Returns the DB row id and image URI."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Insert into DB as blob; minimal metadata filled
    row_id = faces_db.insert_face(
        person_id=person_id,
        track_id=0,
        image_path=None,
        image_blob=content,
        source_video="[upload]",
        frame_number=-1,
        confidence=0.0,
        embedding=None,
        quality_score=0.0,
        is_blurry=False,
        laplacian_var=0.0,
        brightness_ok=True,
        mean_brightness=0.0,
        pose_ok=True,
        yaw=0.0,
        pitch=0.0,
        is_low_res=False,
        width=0,
        height=0,
        is_best_face=False,
        landmarks=None,
    )

    return {"row_id": row_id, "image_uri": f"db://{row_id}", "filename": file.filename}


@app.get("/images")
def list_images(limit: int = 100):
    """List recent images stored in the DB."""
    with faces_db._connect() as conn:
        rows = conn.execute(
            "SELECT id, person_id, image_path, created_at FROM faces ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"id": r["id"], "person_id": r["person_id"], "image_path": r["image_path"], "created_at": r["created_at"]} for r in rows]


@app.get("/images/{row_id}")
def get_image(row_id: int):
    """Return the raw JPEG bytes for a stored image row (if present)."""
    blob = faces_db.get_image_blob(row_id)
    if blob is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=blob, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=8000, reload=True)
