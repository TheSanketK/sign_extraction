import numpy as np

def crop_signature_roi(normalized_image: np.ndarray) -> np.ndarray:
    """
    Extracts the bottom-right signature zone based on fixed percentage coordinates.
    Expects a normalized 2020x930 image.
    """
    H, W = normalized_image.shape[:2]
    
    # X-Axis: Spans the right-most 45% of the cheque
    x_start = int(W * 0.65)
    x_end = W
    
    # Y-Axis: Avoid bottom MICR band (~15%), starts roughly midway down (~45%)
    y_start = int(H * 0.60)
    y_end = int(H * 0.85)
    
    # Crop using NumPy array slicing
    signature_crop = normalized_image[y_start:y_end, x_start:x_end]
    
    return signature_crop