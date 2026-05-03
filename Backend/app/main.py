from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import các module từ project của bạn
from app.api.routes import predict
from app.services.model_service import load_all_models

# --- QUẢN LÝ VÒNG ĐỜI (LIFESPAN) CỦA ỨNG DỤNG ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Những code viết ở đây sẽ chạy 1 LẦN DUY NHẤT khi server khởi động
    print("🚀 Đang khởi động hệ thống Backend AI...")
    
    # Gọi hàm load 6 models (từ RF, XGBoost đến ResNet) vào RAM
    load_all_models()
    
    yield # Bắt đầu cho phép server nhận request từ người dùng
    
    # Những code viết ở đây sẽ chạy khi bạn tắt server (Ctrl + C)
    print("🛑 Đang tắt hệ thống và giải phóng bộ nhớ...")

# --- KHỞI TẠO FASTAPI ---
app = FastAPI(
    title="Hệ Thống Nhận Diện & Đánh Giá Chất Lượng Trái Cây",
    description="Backend API sử dụng Random Forest, XGBoost và ResNet",
    version="1.0.0",
    lifespan=lifespan
)

# --- CẤU HÌNH CORS ---
# Cho phép Frontend (React, Vue, HTML thuần...) gọi được API này mà không bị lỗi chéo domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Trong thực tế production, bạn nên thay "*" bằng domain thật của Frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GẮN CÁC ROUTER ---
# Gắn router dự đoán ảnh vào đường dẫn gốc là /predict
app.include_router(predict.router, prefix="/predict", tags=["AI Predictions"])

# --- ENDPOINT KIỂM TRA SỨC KHỎE SERVER ---
@app.get("/", tags=["Health Check"])
def read_root():
    return {
        "status": "success",
        "message": "Hệ thống Backend CV Multi-Model đang hoạt động! Truy cập http://localhost:8001/docs để test API."
    }