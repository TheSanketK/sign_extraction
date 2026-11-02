# src/controller/deterministic_controller.py
from fastapi import UploadFile, HTTPException, File
from fastapi.responses import Response
from typing import List, Union
import traceback
import io
import zipfile

# Import the orchestrator service we just created
from src.services.sign_services.deterministic_service import deterministic_service

async def process_deterministic_extraction(files: Union[UploadFile, List[UploadFile]] = File(...)) -> Response:
    if not isinstance(files, list):
        files = [files]

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    all_extracted_signatures = []

    try:
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")

            image_bytes = await file.read()
            original_name = file.filename or "document"
            name_without_ext = original_name.rsplit('.', 1)[0]

            try:
                # The deterministic flow returns a single cropped ROI
                crop_bytes = deterministic_service.process_cheque(image_bytes)
                
                file_name = f"{name_without_ext}_det_sig.jpg"
                all_extracted_signatures.append((file_name, crop_bytes))
                    
            except Exception as e:
                print(f"Skipping {original_name}: {str(e)}")
                continue

        if not all_extracted_signatures:
            raise ValueError("No signatures could be extracted using the deterministic method.")

        # SCENARIO A: 1 file -> View INLINE in Swagger
        if len(all_extracted_signatures) == 1:
            download_name = all_extracted_signatures[0][0]
            headers = {"Content-Disposition": f'inline; filename="{download_name}"'}
            
            return Response(
                content=all_extracted_signatures[0][1], 
                media_type="image/jpeg",
                headers=headers
            )

        # SCENARIO B: Multiple files -> Force Download ZIP archive
        else:
            zip_buffer = io.BytesIO()
            used_names = set()
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for file_name, img_bytes in all_extracted_signatures:
                    unique_name = file_name
                    counter = 1
                    while unique_name in used_names:
                        unique_name = file_name.replace(".jpg", f"({counter}).jpg")
                        counter += 1
                    
                    used_names.add(unique_name)
                    zip_file.writestr(unique_name, img_bytes)
            
            zip_bytes = zip_buffer.getvalue()
            download_name = "batch_deterministic_signatures.zip"
            headers = {"Content-Disposition": f'attachment; filename="{download_name}"'}
            
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