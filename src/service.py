import bentoml
import torch
import numpy as np
from bentoml.io import JSON

# 1. Khai báo Service và các Model phụ thuộc ngay trong decorator
@bentoml.service(
    name="titanic_service",
    traffic={"timeout": 60}
)
class TitanicService:
    # BentoML sẽ tự động load model và scaler vào đây
    classifier_model = bentoml.pytorch.get("titanic_classifier:latest")
    scaler_obj = bentoml.picklable_model.get("titanic_scaler:latest")

    def __init__(self):
       # Đăng ký class của bạn vào danh sách an toàn của PyTorch
        from models.model import SimpleClassifier
        torch.serialization.add_safe_globals([SimpleClassifier])
        
        # Bây giờ mới load model
        self.model = self.classifier_model.load_model(weights_only=False)
        self.scaler = self.scaler_obj.load_model()
        self.model.eval()

    @bentoml.api
    async def predict(self, pclass: int, sex: int, age: float, fare: float) -> dict:
        # Xử lý data đầu vào
        raw_input = np.array([[pclass, sex, age, fare]])
        
        # Scale dữ liệu
        scaled_data = self.scaler.transform(raw_input)
        input_tensor = torch.as_tensor(scaled_data, dtype=torch.float32)
        
        # Dự đoán
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output).item()
            
        return {
            "prediction": "Survived" if prediction == 1 else "Deceased"
        }