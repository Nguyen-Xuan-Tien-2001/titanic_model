import torch
import os
import yaml
from models.model import SimpleClassifier

def generate_model():
    # 1. Load config để lấy đúng cấu trúc
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']

    # 2. Khởi tạo model với cấu trúc đã định nghĩa
    model = SimpleClassifier(
        model_cfg['input_size'], 
        model_cfg['hidden_size'], 
        model_cfg['num_classes']
    )

    # 3. Tạo thư mục models nếu chưa có
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Đã tạo thư mục models/")

    # 4. Lưu model (lúc này trọng số là ngẫu nhiên)
    torch.save(model.state_dict(), 'models/my_first_model.pth')
    print("Đã tạo file models/my_first_model.pth thành công!")

if __name__ == "__main__":
    generate_model()