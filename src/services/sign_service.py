import cv2
import numpy as np
import onnxruntime as ort
from io import BytesIO
from PIL import Image, ImageOps
from src.constants.config import DETECTION_MODEL, CONFIDENCE_THRESHOLD, INFERENCE_SIZE

class SignatureService:
    def __init__(self):
        # Configure ONNX Runtime for optimal CPU performance
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Initialize only the Detection session
        self.det_session = ort.InferenceSession(
            DETECTION_MODEL, sess_options, providers=["CPUExecutionProvider"]
        )
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_output_name = self.det_session.get_outputs()[0].name

    def crop_signatures(self, image_bytes: bytes) -> list[bytes]:
        # 1. EXIF Auto-Rotation and Image Decoding
        try:
            img_pil = Image.open(BytesIO(image_bytes))
            img_pil = ImageOps.exif_transpose(img_pil)
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Invalid image file format: {str(e)}")

        orig_h, orig_w = img.shape[:2]

        # 2. Detection Inference Prep
        resized = cv2.resize(img, (INFERENCE_SIZE, INFERENCE_SIZE))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and change shape to (Batch, Channel, Height, Width)
        blob = rgb_img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0

        # 3. Run Inference via ONNX
        outputs = self.det_session.run([self.det_output_name], {self.det_input_name: blob})[0]
        
        # YOLO outputs typically shape (1, num_classes + 4, num_boxes)
        # Transpose to (num_boxes, num_classes + 4)
        predictions = outputs[0].T  
        
        # YOLO format: [cx, cy, w, h, class_0_conf, class_1_conf...]
        # Assuming class 0 is 'signature', index 4 is its confidence score
        scores = predictions[:, 4]
        mask = scores > CONFIDENCE_THRESHOLD
        filtered_preds = predictions[mask]

        if len(filtered_preds) == 0:
            raise ValueError("No signatures detected in this document.")

        # 4. Scale Coordinates back to original image size
        x_scale = orig_w / INFERENCE_SIZE
        y_scale = orig_h / INFERENCE_SIZE

        cx, cy = filtered_preds[:, 0], filtered_preds[:, 1]
        w_norm, h_norm = filtered_preds[:, 2], filtered_preds[:, 3]
        conf = filtered_preds[:, 4]

        # Convert Center-XY to Top-Left-XY and scale width/height
        x1 = (cx - (w_norm / 2)) * x_scale
        y1 = (cy - (h_norm / 2)) * y_scale
        w = w_norm * x_scale
        h = h_norm * y_scale

        boxes_xywh = np.column_stack([x1, y1, w, h])

        # 5. Non-Maximum Suppression (Remove overlapping duplicate boxes)
        nms_indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), conf.tolist(), CONFIDENCE_THRESHOLD, 0.45 
        )

        if len(nms_indices) == 0:
            raise ValueError("No signatures detected in this document.")

        # 6. Extract ALL valid signatures
        cropped_signatures_list = []
        
        for i in np.ravel(nms_indices):
            bx, by, bw, bh = boxes_xywh[i]

            # Add 20% padding around the signature so it's not cropped too tightly
            pad_x, pad_y = int(bw * 0.20), int(bh * 0.20)
            
            crop_x1 = max(0, int(bx - pad_x))
            crop_y1 = max(0, int(by - pad_y))
            crop_x2 = min(orig_w, int(bx + bw + pad_x))
            crop_y2 = min(orig_h, int(by + bh + pad_y))
            
            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

            success, buffer = cv2.imencode(".jpg", cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if success:
                cropped_signatures_list.append(buffer.tobytes())

        if not cropped_signatures_list:
            raise RuntimeError("Failed to encode cropped images")
        return cropped_signatures_list

sign_service = SignatureService()