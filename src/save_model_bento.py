import bentoml
import torch
import joblib
import yaml
from models.model import SimpleClassifier

# 1. Load config để biết kích thước Model
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
m_cfg = config['model']

# 2. Khởi tạo thực thể Model trắng
model = SimpleClassifier(m_cfg['input_size'], m_cfg['hidden_size'], m_cfg['num_classes'])

# 3. Nạp trọng số (state_dict) từ file .pth vào Model
model_weights = torch.load('models/titanic_model.pth')
model.load_state_dict(model_weights)
model.eval() # Chuyển sang chế độ dự đoán

# 4. Lưu Model vào BentoML Store
# Sử dụng signatures để bật tính năng Adaptive Batching sau này
bentoml.pytorch.save_model(
    "titanic_classifier", 
    model,
    signatures={"__call__": {"batchable": True}}
)

# 5. Lưu Scaler (vẫn dùng picklable_model như cũ)
scaler = joblib.load('models/scaler.joblib')
bentoml.picklable_model.save_model("titanic_scaler", scaler)

print("✅ Đã đăng ký thành công Model (Module) và Scaler vào BentoML Store!")