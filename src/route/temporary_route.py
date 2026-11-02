from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from typing import List, Union

from src.controller.temporary_controller import process_visualize_detection
from src.schema.sign_schema import ErrorResponse

router = APIRouter(prefix="/api/v1/signatures")

@router.post(
    "/detect/visualize",
    response_class=Response,
    tags=["Temporary Visualizer"],
    responses={
        200: {
            "description": "Returns a full image with bounding boxes drawn over detected signatures.",
            "content": {"image/jpeg": {}, "application/zip": {}}
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Draw boxes around detected signatures"
)
async def visualize_signature(files: Union[UploadFile, List[UploadFile]] = File(...)):
    """
    Demonstration endpoint: Runs YOLO11 detection and returns the original document
    with green bounding boxes outlining the detected signature zones.
    """
    return await process_visualize_detection(files)