import cv2
import numpy as np

def normalize_cheque_size(image: np.ndarray) -> np.ndarray:
    """
    Resizes the flattened cheque to a fixed standard resolution of 2020 x 930 pixels.
    This matches the Indian CTS-2010 standard physical dimensions (202 mm x 93 mm).
    """
    TARGET_WIDTH = 2020
    TARGET_HEIGHT = 930
    
    # INTER_AREA is best for downsampling, INTER_CUBIC is best for upsampling.
    # We use INTER_CUBIC as a safe default for high-quality resizing.
    normalized = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    
    return normalized