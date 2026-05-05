import pandas as pd
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_titanic_data(save_scaler=False, scaler_path='models/scaler.joblib'):
    # 1. Tải dữ liệu từ thư viện
    df = sns.load_dataset('titanic')
    
    # 2. Chọn các cột (Features) quan trọng
    # Chúng ta chọn: Hạng vé (pclass), Giới tính (sex), Tuổi (age), Giá vé (fare)
    # Mục tiêu (Target): Sống sót (survived)
    cols = ['pclass', 'sex', 'age', 'fare', 'survived']
    df = df[cols]
    
    # 3. Xử lý dữ liệu thiếu (Data Cleaning)
    # Cột 'age' thường bị thiếu, ta điền giá trị trung bình vào
    df['age'] = df['age'].fillna(df['age'].mean())
    
    # 4. Chuyển đổi chữ thành số (Encoding)
    # Máy tính không hiểu "male"/"female", ta chuyển thành 0/1
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    
    scaler = StandardScaler()
    # Chuẩn hóa 4 cột đầu vào
    df[['pclass', 'sex', 'age', 'fare']] = scaler.fit_transform(df[['pclass', 'sex', 'age', 'fare']])

    if save_scaler:
        scaler_dir = os.path.dirname(scaler_path)
        if scaler_dir:
            os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return df

if __name__ == "__main__":
    data = load_titanic_data()
    print("Dữ liệu sau khi xử lý:")
    print(data.head())