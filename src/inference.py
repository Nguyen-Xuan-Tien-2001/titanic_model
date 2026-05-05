import torch
from models.model import SimpleClassifier
import yaml
import torch.nn.functional as F
import joblib
import os

def predict():
    # 1. Load cấu hình (để lấy đúng INPUT_SIZE, HIDDEN_SIZE...)
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']

    # 2. Khởi tạo cấu trúc model và load trọng số đã train (weights)
    model = SimpleClassifier(
        model_cfg['input_size'], 
        model_cfg['hidden_size'], 
        model_cfg['num_classes']
    )
    
    # Giả sử bạn đã chạy train.py và có file .pth trong folder models/
    try:
        model.load_state_dict(torch.load('models/titanic_model.pth'))
        model.eval() # Chuyển sang chế độ dự đoán (quan trọng!)
        print("Đã load model thành công.")
    except FileNotFoundError:
        print("Chưa tìm thấy file model.pth. Bạn cần chạy train.py trước!")
        return

    # 3. Giả lập một dữ liệu đầu vào (ví dụ: 1 mẫu dữ liệu có 10 chỉ số)
    # Trong thực tế, đây có thể là dữ liệu từ API hoặc User nhập vào
    # sample_input = torch.randn(1, model_cfg['input_size']) 
    # sample_input = torch.zeros(1, model_cfg['input_size'])
    # pclass=3, sex=1, age=22, fare=7.25
    # sample_input = torch.tensor([[3, 1, 22.0, 7.25]], dtype=torch.float32)
    raw_input = [[1, 0, 38.0, 71.28]]
    scaler_path = 'models/scaler.joblib'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        transformed_input = scaler.transform(raw_input)
        sample_input = torch.tensor(transformed_input, dtype=torch.float32)
    else:
        print("Không tìm thấy scaler.joblib. Dùng dữ liệu thô, kết quả có thể không chính xác.")
        sample_input = torch.tensor(raw_input, dtype=torch.float32)

    # 4. Thực hiện dự đoán
    with torch.no_grad(): # Tắt tính toán gradient để tiết kiệm RAM/CPU
        logits = model(sample_input)
        probabilities = F.softmax(logits, dim=1)
        # 3. Lấy ra class có xác suất cao nhất
        confidence, predicted_class = torch.max(probabilities, dim=1)

        prediction = model(sample_input)
        predicted_class = torch.argmax(prediction, dim=1)
    
    print(f"--- KẾT QUẢ DỰ ĐOÁN ---")
    print(f"Class: {predicted_class.item()}")
    print(f"Độ tự tin: {confidence.item() * 100:.2f}%")
    
    # In chi tiết xác suất của tất cả các class
    for i, prob in enumerate(probabilities[0]):
        print(f"  > Xác suất Class {i}: {prob.item() * 100:.2f}%")

if __name__ == "__main__":
    predict()