import cv2
import numpy as np
from src.services.sign_services.sign_preprocessing import deskew_cheque
from src.services.sign_services.sign_normalize import normalize_cheque_size
from src.services.sign_services.sign_roi_extractor import crop_signature_roi

# Import your utility file here (adjust the function name if you named it differently)
from src.services.sign_services.convert_to_greyscale import bytes_to_grayscale

class DeterministicSignService:
    def process_cheque(self, image_bytes: bytes) -> bytes:
        """
        Executes the deterministic OpenCV pipeline: Grayscale -> Deskew -> Normalize -> Crop ROI.
        """
        # 1. Decode bytes directly to Grayscale using your utility!
        try:
            gray_img = bytes_to_grayscale(image_bytes)
        except Exception:
            # Fallback in case your utility expects numpy arrays instead of bytes
            nparr = np.frombuffer(image_bytes, np.uint8)
            gray_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if gray_img is None:
            raise ValueError("Invalid image format provided to deterministic flow.")

        # 2. Step 1: Preprocess & Deskew (Now operates on and returns grayscale)
        deskewed_img = deskew_cheque(gray_img)

        # 3. Step 2: Normalize to 2020x930
        normalized_img = normalize_cheque_size(deskewed_img)

        # 4. Step 3: Extract fixed ROI
        roi_img = crop_signature_roi(normalized_img)

        # 5. Encode output back to bytes
        success, buffer = cv2.imencode(".jpg", roi_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not success:
            raise RuntimeError("Failed to encode deterministic ROI crop.")

        return buffer.tobytes()

deterministic_service = DeterministicSignService()