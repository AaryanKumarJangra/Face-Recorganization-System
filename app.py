from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class VideoRequest(BaseModel):
    video_url: str

# Placeholder import for your existing logic. Replace with actual function.
from training.dataset_builder import build_dataset  # example placeholder

def run_face_recognition(video_url: str):
    # TODO: Replace this with your repo's detection/recognition pipeline call
    # Example placeholder implementation — adapt to your project's API
    return {"message": f"processed {video_url} (placeholder)"}

@app.post("/process-video")
async def process_video(request: VideoRequest):
    result = run_face_recognition(request.video_url)
    return {"status": "success", "result": result}

@app.get("/")
def health_check():
    return {"status": "running"}
