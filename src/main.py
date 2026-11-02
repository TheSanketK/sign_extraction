# src/main.py
from fastapi import FastAPI
from src.route.sign_route import router as sign_router 
from src.route.temporary_route import router as temporary_router 
from src.constants.config import APP_HOST, APP_PORT

app = FastAPI(
    title="Signature Detection API",
    description="An API to detect and annotate signatures in document images using YOLO11n.",
    version="1.0.0"
)

app.include_router(sign_router)
app.include_router(temporary_router)

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "Signature Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    # Now uses the port and host from your .env file
    uvicorn.run("src.main:app", host=APP_HOST, port=APP_PORT, reload=True)