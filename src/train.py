import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from models.model import SimpleClassifier
from src.data_loader import load_titanic_data

def train():
    # 1. Load Config
    # Lấy dữ liệu thật
    df = load_titanic_data(save_scaler=True)
    # Chuyển từ Pandas sang Tensor của PyTorch
    X = torch.tensor(df[['pclass', 'sex', 'age', 'fare']].values, dtype=torch.float32)
    y = torch.tensor(df['survived'].values, dtype=torch.long)

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    m_cfg = config['model']
    t_cfg = config['training']

    # 2. Tạo dữ liệu giả lập (Dummy Data) để học
    # Tạo 10000 mẫu, mỗi mẫu có 10 đặc trưng (input_size)
    # X = torch.randn(10000, m_cfg['input_size'])
    # Logic: Nếu tổng hàng > 0 thì nhãn là 1, ngược lại là 0
    # y = (X.sum(dim=1) > 0).long() 

    # 3. Khởi tạo Model, Loss, Optimizer
    model = SimpleClassifier(m_cfg['input_size'], m_cfg['hidden_size'], m_cfg['num_classes'])
    
    # --- PHẦN LOGIC RETRAIN ---
    model_path = 'models/titanic_model.pth'
    if os.path.exists(model_path):
        print(f"Tìm thấy model cũ tại {model_path}. Đang load để học tiếp...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Không tìm thấy model cũ. Bắt đầu học từ đầu (ngẫu nhiên).")
    # --------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=t_cfg['learning_rate'])

    # 4. Vòng lặp huấn luyện
    print("Bắt đầu huấn luyện...")
    model.train()
    for epoch in range(t_cfg['epochs']):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Tính Accuracy
        with torch.no_grad():
            # Lấy index của xác suất cao nhất
            predicted = torch.argmax(outputs, dim=1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
            
        print(f"Epoch [{epoch+1}/{t_cfg['epochs']}], Loss: {loss.item():.4f}, Acc: {accuracy*100:.2f}%")

    # 5. Lưu model đã "khôn" hơn vào thư mục models/
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), 'models/titanic_model.pth')
    print("Huấn luyện xong & Đã lưu model!")

if __name__ == "__main__":
    train()