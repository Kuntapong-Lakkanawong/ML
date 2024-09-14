import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# โหลดโมเดล
model = load_model("model.h5")

# โหลด scaler สำหรับการ inverse transform
scaler_y = pickle.load(open("scaler_y.pkl", "rb"))  # ต้องแน่ใจว่าคุณบันทึก scaler_y ตอนที่ทำการเทรนโมเดล

st.title('Gold Price Prediction')

# รับค่าฟีเจอร์จากผู้ใช้
feature1 = st.number_input('open')
feature2 = st.number_input('high')
feature3 = st.number_input('low')
feature4 = st.number_input('rsi')
feature5 = st.number_input('sma')

# แสดงปุ่มสำหรับทำนาย
if st.button('Predict'):
    # รวบรวมฟีเจอร์ทั้งหมดเป็น array เพื่อส่งเข้าโมเดล
    data = np.array([[feature1, feature2, feature3, feature4, feature5]])
    prediction = model.predict(data)
    
    # ทำ inverse scaling เพื่อแปลงค่า prediction กลับเป็นราคาจริง
    predicted_price = scaler_y.inverse_transform(prediction)
    
    # แสดงผลลัพธ์
    st.write(f'The predicted closing price is: {predicted_price[0][0]}')