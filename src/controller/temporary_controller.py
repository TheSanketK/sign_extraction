from fastapi import UploadFile, HTTPException, File
from fastapi.responses import Response
from typing import List, Union
from src.services.temporary_sign_service import temporary_sign_service
import traceback
import io
import zipfile

async def process_visualize_detection(files: Union[UploadFile, List[UploadFile]] = File(...)) -> Response:
    if not isinstance(files, list):
        files = [files]

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    all_drawn_images = [] 

    try:
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")

            image_bytes = await file.read()
            original_name = file.filename or "document"
            name_without_ext = original_name.rsplit('.', 1)[0]

            try:
                # Returns the full image with boxes drawn on it
                drawn_image_bytes = temporary_sign_service.draw_signature_boxes(image_bytes)
                file_name = f"{name_without_ext}_visualized.jpg"
                all_drawn_images.append((file_name, drawn_image_bytes))
                    
            except ValueError:
                continue

        if not all_drawn_images:
            raise ValueError("No signatures detected in any of the uploaded documents.")

        # Single file -> View INLINE
        if len(all_drawn_images) == 1:
            download_name = all_drawn_images[0][0]
            # headers = {"Content-Disposition": f'inline; filename="{download_name}"'}
            headers = {"Content-Disposition": f'attachment; filename="{download_name}"'}
            return Response(
                content=all_drawn_images[0][1], 
                media_type="image/jpeg",
                headers=headers
            )

        # Batch files -> Download ZIP
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for file_name, img_bytes in all_drawn_images:
                    zip_file.writestr(file_name, img_bytes)
            
            zip_bytes = zip_buffer.getvalue()
            download_name = "batch_visualized_signatures.zip"
            headers = {"Content-Disposition": f'attachment; filename="{download_name}"'}
            
            return Response(
                content=zip_bytes, 
                media_type="application/zip",
                headers=headers
            )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))