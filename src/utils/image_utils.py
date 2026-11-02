# src/utils/image_utils.py
import cv2
import numpy as np
from typing import Union

def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Converts an OpenCV BGR image array to grayscale.
    If the image is already grayscale, it returns it untouched.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Invalid image array provided.")
        
    # Check if the image is already grayscale (2 dimensions)
    if len(img.shape) == 2:
        return img
        
    # Convert BGR (OpenCV default) to Grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def bytes_to_grayscale(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes directly into a grayscale OpenCV array.
    This is faster for API routes because it skips the BGR decoding step entirely.
    """
    if not image_bytes:
        raise ValueError("Empty image bytes provided.")

    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # IMREAD_GRAYSCALE natively decodes the file directly into a single-channel image
    gray_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if gray_img is None:
        raise ValueError("Failed to decode image bytes. Format may be unsupported.")
        
    return gray_img