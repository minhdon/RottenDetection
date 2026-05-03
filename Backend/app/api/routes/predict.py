from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.services.model_service import predict_image

router = APIRouter()

@router.post("/upload")
async def predict_upload(
    file: UploadFile = File(...),
    # Thêm tham số để chọn pipeline mô hình. Mặc định là resnet_xgb
    model_type: str = Form("resnet_xgb") 
):
    # Các model_type hợp lệ: "rf_rgb", "xgb_rgb", "resnet_xgb", "resnet_rf"
    valid_models = ["rf_rgb", "xgb_rgb", "resnet_xgb","resnet_rf"]
    if model_type not in valid_models:
        raise HTTPException(status_code=400, detail=f"model_type phải thuộc {valid_models}")

    try:
        image_bytes = await file.read()
        
        # Gọi service xử lý, truyền thêm model_type
        result = predict_image(image_bytes, model_type)
        
        return {
            "filename": file.filename,
            "pipeline_used": model_type,
            "predictions": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")