import torch
import joblib
import yaml
from fastapi import FastAPI, Depends, HTTPException, Request, status
from pydantic import BaseModel
from models.model import SimpleClassifier
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

import bcrypt 
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta


SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    

# Khởi tạo FastAPI

app = FastAPI(title="Titanic Survival Prediction API")

# --- 1. SETUP MODEL & SCALER ---
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
m_cfg = config['model']

model = SimpleClassifier(m_cfg['input_size'], m_cfg['hidden_size'], m_cfg['num_classes'])
model.load_state_dict(torch.load('models/titanic_model.pth'))
model.eval()
scaler = joblib.load('models/scaler.joblib')

# Định nghĩa Schema cho Request
class Passenger(BaseModel):
    pclass: int
    sex: int
    age: float
    fare: float

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Endpoint bất đồng bộ
@app.post("/predict")
@limiter.limit("5/minute")
async def predict( request: Request, passenger: Passenger ,current_user: str = Depends(get_current_user)):
    try:
        # Xử lý data
        raw_data = [[passenger.pclass, passenger.sex, passenger.age, passenger.fare]]
        scaled_data = scaler.transform(raw_data)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # Inference (Dự đoán)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, dim=1)

        return {
            "prediction": "Survived" if pred.item() == 1 else "Deceased",
            "confidence": f"{conf.item()*100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
 

# Tạo sẵn hash cho mật khẩu "admin123"
# Lưu ý: Bcrypt yêu cầu dữ liệu dạng bytes
password_bytes = "admin123".encode('utf-8')
hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": hashed.decode('utf-8'), # Lưu dưới dạng chuỗi để dễ đọc
    }
}

# 3. Endpoint để lấy Token (Login)
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user:
        raise HTTPException(status_code=400, detail="Sai username hoặc password")
    
    # Kiểm tra mật khẩu dùng trực tiếp thư viện bcrypt
    password_byte = form_data.password.encode('utf-8')
    hashed_byte = user["hashed_password"].encode('utf-8')
    
    if not bcrypt.checkpw(password_byte, hashed_byte):
        raise HTTPException(status_code=400, detail="Sai username hoặc password")
    
    # Phần tạo JWT Token bên dưới giữ nguyên...
    access_token_expires = timedelta(minutes=30)
    access_token = jwt.encode(
        {"sub": user["username"], "exp": datetime.utcnow() + access_token_expires}, 
        SECRET_KEY, 
        algorithm=ALGORITHM
    )
    return {"access_token": access_token, "token_type": "bearer"}