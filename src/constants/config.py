import os
from dotenv import load_dotenv

load_dotenv(override=True)
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))

DETECTION_MODEL = os.getenv("DETECTION_MODEL")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.4))
DEVICE = os.getenv("DEVICE", "cpu")
INFERENCE_SIZE = int(os.getenv("INFERENCE_SIZE"))
print(f"inference size: {INFERENCE_SIZE}")