import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import pathlib
import io
import joblib 
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
# Dictionary lưu trữ tất cả mô hình trên RAM
MODELS = {
    "rf_fruit": None, "rf_quality": None,
    "xgb_fruit": None, "xgb_quality": None,
    "resnet_extractor": None,
    "resnet_xgb_fruit": None, "resnet_xgb_quality": None,
    "resnet_rf_fruit": None, "resnet_rf_quality": None
}

# --- CÁC HÀM TIỀN XỬ LÝ ---

def extract_rgb_features(img_pil):
    """
    Hàm rút trích đặc trưng bằng Color Histogram (256 bins * 3 kênh = 768 features)
    Đã được đồng bộ 100% với file huấn luyện.
    """
    # img_pil đã được convert("RGB") ở hàm gọi nó, nên mảng Numpy cũng chuẩn RGB
    img_np = np.array(img_pil)
    
    bins = 256
    feat_vector = []
    
    # Tính Histogram cho 3 kênh màu (R, G, B)
    for i in range(3):
        # OpenCV yêu cầu đầu vào phải là numpy array
        hist = cv2.calcHist([img_np], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feat_vector.extend(hist)
        
    feat_vector = np.array(feat_vector)
    feat_vector = cv2.normalize(feat_vector, feat_vector).flatten()
    
    # Random Forest / XGBoost yêu cầu đầu vào mảng 2D (1 sample, n_features)
    return np.array([feat_vector])

def transform_for_resnet(img_pil):
    """Tiền xử lý ảnh cho ResNet PyTorch"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img_pil).unsqueeze(0)


# --- HÀM LOAD TẤT CẢ MÔ HÌNH ---

def load_all_models():
    """Chạy 1 lần khi khởi động FastAPI"""
    print("⏳ Đang nạp toàn bộ 6 mô hình vào bộ nhớ...")
    current_dir = pathlib.Path(__file__).resolve().parent
    models_dir = current_dir.parent / 'models'
    
    try:
        # 1. Load các model Machine Learning truyền thống
        MODELS["rf_fruit"] = joblib.load(models_dir / 'rf_rgb_fruit_model.pkl')
        print("   -> Đã nạp RF Fruit")
        MODELS["rf_quality"] = joblib.load(models_dir / 'rf_rgb_quality_model.pkl')
        print("   -> Đã nạp RF Quality")
        MODELS["xgb_fruit"] = joblib.load(models_dir / 'xgb_rgb_fruit_model.pkl')
        print("   -> Đã nạp XGB Fruit")
        MODELS["xgb_quality"] = joblib.load(models_dir / 'xgb_rgb_quality_model.pkl')
        print("   -> Đã nạp XGB Quality")
        
        # 2. Load model ResNet PyTorch
        resnet_path = models_dir / 'resnet_xgb_final_backbone.pt'
        print(f"   -> Đang nạp ResNet từ: {resnet_path}")
        
        device = torch.device('cpu') 
        
        # Tạo bộ khung ResNet18 chuẩn và bỏ lớp FC cuối cùng
        resnet = models.resnet18() 
        resnet.fc = nn.Identity()
        
        # Xử lý sự khác biệt tên lớp của FastAI vs PyTorch
        fastai_state_dict = torch.load(resnet_path, map_location=device, weights_only=True)
        pytorch_state_dict = OrderedDict()
        
        for k, v in fastai_state_dict.items():
            if k.startswith('base.'):
                new_key = k[5:] # Gọt bỏ chữ 'base.'
                
                # Sửa tên các lớp ngoài cùng
                if new_key == 'conv2.weight': 
                    new_key = 'conv1.weight'
                elif new_key.startswith('bn.'): 
                    new_key = new_key.replace('bn.', 'bn1.')
                
                # Sửa tên các block bên trong ResNet
                new_key = new_key.replace('.conv1.0.weight', '.conv1.weight')
                new_key = new_key.replace('.conv1.1.', '.bn1.')
                new_key = new_key.replace('.conv2.0.weight', '.conv2.weight')
                new_key = new_key.replace('.conv2.1.', '.bn2.')
                
                pytorch_state_dict[new_key] = v

        # Nạp trọng số vào khung
        resnet.load_state_dict(pytorch_state_dict, strict=False)
        resnet.eval() 
        MODELS["resnet_extractor"] = resnet
        print("   -> Đã nạp ResNet Backbone")
        
        # 3. Load XGBoost Heads cho ResNet
        MODELS["resnet_xgb_fruit"] = joblib.load(models_dir / 'resnet_xgb_final_fruit_head.pkl')
        print("   -> Đã nạp ResNet-XGB Fruit Head")
        MODELS["resnet_xgb_quality"] = joblib.load(models_dir / 'resnet_xgb_final_quality_head.pkl')
        print("   -> Đã nạp ResNet-XGB Quality Head")
        MODELS["resnet_rf_fruit"] = joblib.load(models_dir / 'resnet_rf_final_fruit_head.pkl')
        print("   -> Đã nạp ResNet-RF Fruit Head")
        MODELS["resnet_rf_quality"] = joblib.load(models_dir / 'resnet_rf_final_quality_head.pkl')
        print("   -> Đã nạp ResNet-RF Quality Head")
        
        print("✅ Đã nạp thành công TẤT CẢ mô hình!")
    except Exception as e:
        print(f"❌ Lỗi nạp mô hình: {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"Lỗi ở dòng: {exc_tb.tb_lineno}")

# --- HÀM DỰ ĐOÁN ĐIỀU PHỐI (ROUTER) ---

def predict_image(image_bytes: bytes, model_type: str):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    result = {
        "fruit_type": None,
        "quality": None
    }
    
    # LUỒNG 1: Sử dụng Random Forest
    if model_type == "rf_rgb":
        features = extract_rgb_features(img_pil)
        result["fruit_type"] = MODELS["rf_fruit"].predict(features)[0]
        result["quality"] = MODELS["rf_quality"].predict(features)[0]
        
    # LUỒNG 2: Sử dụng XGBoost
    elif model_type == "xgb_rgb":
        features = extract_rgb_features(img_pil)
        result["fruit_type"] = MODELS["xgb_fruit"].predict(features)[0]
        result["quality"] = MODELS["xgb_quality"].predict(features)[0]
        
    # LUỒNG 3: Pipeline Deep Learning kết hợp (ResNet -> XGBoost)
    elif model_type == "resnet_xgb":
        # Đưa ảnh qua ResNet để lấy đặc trưng
        tensor_img = transform_for_resnet(img_pil)
        with torch.no_grad():
            features_tensor = MODELS["resnet_extractor"](tensor_img)
            features_np = features_tensor.cpu().numpy()
            
        # Đưa feature vector vào XGBoost dự đoán
        result["fruit_type"] = MODELS["resnet_xgb_fruit"].predict(features_np)[0]
        result["quality"] = MODELS["resnet_xgb_quality"].predict(features_np)[0]
    elif model_type == "resnet_rf":
        # Đưa ảnh qua ResNet để lấy đặc trưng
        tensor_img = transform_for_resnet(img_pil)
        with torch.no_grad():
            features_tensor = MODELS["resnet_extractor"](tensor_img)
            features_np = features_tensor.cpu().numpy()
            
        # Đưa feature vector vào Random Forest dự đoán
        result["fruit_type"] = MODELS["resnet_rf_fruit"].predict(features_np)[0]
        result["quality"] = MODELS["resnet_rf_quality"].predict(features_np)[0]

    # Xử lý định dạng kiểu dữ liệu cho JSON (tránh lỗi NumPy type)
    if isinstance(result["fruit_type"], np.generic):
        result["fruit_type"] = result["fruit_type"].item()
    if isinstance(result["quality"], np.generic):
        result["quality"] = result["quality"].item()

    return result