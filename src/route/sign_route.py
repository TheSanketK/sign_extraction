# src/route/sign_route.py
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from typing import List, Union

from src.controller.sign_controller import process_signature_detection
from src.controller.deterministic_controller import process_deterministic_extraction
from src.schema.sign_schema import ErrorResponse

# Removed global tags so we can separate them at the endpoint level
router = APIRouter(prefix="/api/v1/signatures")

# --- NON-DETERMINISTIC (AI FLOW) ---
@router.post(
    "/detect/ai",
    response_class=Response,
    tags=["Non-Deterministic"],
    responses={
        200: {
            "description": "Returns a cropped JPEG (single upload) or ZIP file (batch upload)",
            "content": {"image/jpeg": {}, "application/zip": {}}
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="AI Model Signature Extraction"
)
async def detect_signature_ai(files: Union[UploadFile, List[UploadFile]] = File(...)):
    """
    Uses YOLO11 object detection to intelligently find and crop signatures.
    Supports single or batch image uploads.
    """
    return await process_signature_detection(files)


# --- DETERMINISTIC (OPENCV FLOW) ---
@router.post(
    "/detect/deterministic",
    response_class=Response,
    tags=["Deterministic"],
    responses={
        200: {
            "description": "Returns a cropped JPEG (single upload) or ZIP file (batch upload)",
            "content": {"image/jpeg": {}, "application/zip": {}}
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Heuristic ROI Signature Extraction"
)
async def detect_signature_deterministic(files: Union[UploadFile, List[UploadFile]] = File(...)):
    """
    Uses OpenCV contour detection, perspective warping, and fixed ROI 
    coordinates to extract the bottom-right signature zone of an Indian cheque.
    Supports single or batch image uploads.
    """
    return await process_deterministic_extraction(files)