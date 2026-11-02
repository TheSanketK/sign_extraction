# import cv2
# import numpy as np

# def order_points(pts):
#     """Orders points in: top-left, top-right, bottom-right, bottom-left"""
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
    
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#     return rect

# def deskew_cheque(image: np.ndarray) -> np.ndarray:
#     """Finds the cheque contour and applies a perspective warp to flatten it."""
#     orig_h, orig_w = image.shape[:2]
#     img_area = orig_h * orig_w
    
#     # SAFETY CHECK: If image is already grayscale (2 dimensions), don't convert it again
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()

#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(gray, 50, 150)

#     # Find contours
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not contours:
#         return image 

#     # Sort by area (largest first)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
#     doc_cnt = None
#     for c in contours:
#         # Ignore the contour if it is smaller than 30% of the total image area.
#         if cv2.contourArea(c) < (img_area * 0.3):
#             continue
            
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
#         if len(approx) == 4:
#             doc_cnt = approx
#             break

#     if doc_cnt is None:
#         return image

#     # Apply Perspective Transform
#     rect = order_points(doc_cnt.reshape(4, 2))
#     (tl, tr, br, bl) = rect

#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     maxWidth = max(int(widthA), int(widthB))

#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)
#     maxHeight = max(int(heightA), int(heightB))

#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]
#     ], dtype="float32")

#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

#     return warped

# =====================================================
# src/services/sign_services/sign_preprocessing.py
import cv2
import numpy as np

def order_points(pts):
    """Orders points in: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def deskew_cheque(image: np.ndarray) -> np.ndarray:
    """
    Dynamically finds the cheque against any background, un-rotates it, 
    un-skews it, and crops it flat.
    """
    orig_h, orig_w = image.shape[:2]
    img_area = orig_h * orig_w
    
    # 1. Grayscale Check
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Blur to remove paper texture noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Auto-Canny Edge Detection
    # Automatically calculates thresholds based on the image's median brightness.
    # This prevents failure on very dark or very light backgrounds.
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray, lower, upper)

    # 4. Dilate the edges slightly to close any broken lines around the cheque border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    # 5. Find Contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image 

    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    doc_cnt = None
    for c in contours:
        # Lowered threshold to 15% to handle cheques on very large backgrounds
        if cv2.contourArea(c) < (img_area * 0.15):
            continue
            
        # 6. Dynamic Polygon Approximation
        # Loop through strictness levels to ensure we find exactly 4 corners
        peri = cv2.arcLength(c, True)
        for eps in np.linspace(0.01, 0.05, 10):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
                
        if doc_cnt is not None:
            break

    # Fallback if no valid 4-sided document is found (assume image is already cropped)
    if doc_cnt is None:
        return image

    # 7. Apply Master Transformation
    # This single math block perfectly flattens 3D skew AND 2D rotation.
    rect = order_points(doc_cnt.reshape(4, 2))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped