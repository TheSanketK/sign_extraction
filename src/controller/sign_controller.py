# controller for just cropped image extraction
# src/controller/sign_controller.py
from fastapi import UploadFile, HTTPException, File
from fastapi.responses import Response
from typing import List, Union
from src.services.sign_service import sign_service
import traceback
import io
import zipfile

# Accept either a single UploadFile or a List of UploadFiles for batch processing
async def process_signature_detection(files: Union[UploadFile, List[UploadFile]] = File(...)) -> Response:
    
    # --- DEFENSIVE FIX ---
    # If FastAPI passes a single file object, wrap it in a list so our loop works perfectly.
    if not isinstance(files, list):
        files = [files]

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    all_extracted_signatures = []  # Will hold tuples of (filename, image_bytes)

    try:
        # Loop through every uploaded image
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")

            image_bytes = await file.read()
            original_name = file.filename or "document"
            name_without_ext = original_name.rsplit('.', 1)[0]

            try:
                # 1. AI Model: Find and crop all signatures from the full page
                cropped_images_list = sign_service.crop_signatures(image_bytes)
                
                # 2. Append the raw crops directly (No enhancement/artifact removal applied)
                for idx, crop_bytes in enumerate(cropped_images_list):
                    
                    # Tag each raw signature with its parent document's name
                    file_name = f"{name_without_ext}_sig_{idx + 1}.jpg"
                    all_extracted_signatures.append((file_name, crop_bytes))
                    
            except ValueError:
                # If no signature is found in ONE specific image, skip it and continue
                continue

        # Check if we found ANY signatures across all images
        if not all_extracted_signatures:
            raise ValueError("No signatures detected in any of the uploaded documents.")

        # SCENARIO A: Exactly 1 signature found in total -> View INLINE in Swagger
        if len(all_extracted_signatures) == 1:
            download_name = all_extracted_signatures[0][0]
            
            # CHANGED TO INLINE: This renders the image on the screen in Swagger/Browsers
            # headers = {"Content-Disposition": f'attachment; filename="{download_name}"'}
            headers = {"Content-Disposition": f'inline; filename="{download_name}"'}
            
            return Response(
                content=all_extracted_signatures[0][1], 
                media_type="image/jpeg",
                headers=headers
            )

        # SCENARIO B: Multiple signatures found -> Force Download ZIP archive
        else:
            zip_buffer = io.BytesIO()
            used_names = set()
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for file_name, img_bytes in all_extracted_signatures:
                    unique_name = file_name
                    counter = 1
                    
                    # Prevent file overwrite if two files have the exact same name
                    while unique_name in used_names:
                        unique_name = file_name.replace(".jpg", f"({counter}).jpg")
                        counter += 1
                    
                    used_names.add(unique_name)
                    zip_file.writestr(unique_name, img_bytes)
            
            zip_bytes = zip_buffer.getvalue()
            download_name = "batch_extracted_signatures.zip"
            
            # KEPT AS ATTACHMENT: Browsers cannot render ZIP files inline.
            headers = {"Content-Disposition": f'inline; filename="{download_name}"'}
            
            return Response(
                content=zip_bytes, 
                media_type="application/zip",
                headers=headers
            )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print("\n--- CRASH REPORT ---")
        traceback.print_exc()
        print("--------------------\n")
        raise HTTPException(status_code=500, detail=str(e))