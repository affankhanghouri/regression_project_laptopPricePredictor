import streamlit as st
import pickle
import numpy as np

# Load trained model and dataset
pipe = pickle.load(open('pipe2.pkl', 'rb'))
data = pickle.load(open('data2.pkl', 'rb'))

st.title('Laptop Price Predictor')

# Input Fields
brand = st.selectbox('Brand', data['Company'].unique())
type_name = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight (kg)')
touchscreen = st.selectbox('Touchscreen', ['Yes', 'No'])
screen_size = st.number_input('Screen Size (in Inches)', min_value=10.0, max_value=20.0, step=0.1)

ips = st.selectbox('IPS Display', ['Yes', 'No'])
resolution = st.selectbox('Screen Resolution', ['1920 x 1080', '1366 x 768', '1600 x 900', '3840 x 2160', '2560 x 1600', '2560 x 1440'])
cpu = st.selectbox('CPU', data['Cpu_Category'].unique())
gpu = st.selectbox('GPU', data['Gpu_category'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
flashStorage=st.selectbox('flash storage (in GB)',[0,32,64,256,16,128,512])


ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
os = st.selectbox('Operating System', data['categorize_opsys'].unique())

# Extract PPI (Pixels Per Inch) from resolution
X_res = int(resolution.split('x')[0])
y_res = int(resolution.split('x')[1])


ppi = ((X_res**2 + y_res**2) ** 0.5) / (screen_size)  # Ensure screen size is provided

# Convert categorical to binary
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

if st.button('Predict Price'):
    # Ensure all 14 features are included
    hybrid=514
    query = np.array([brand, type_name, ram, weight, touchscreen, ips, ppi, cpu, ssd, hdd, flashStorage,hybrid,gpu, os]).reshape(1, 14)
    
    predicted_price = pipe.predict(query)[0]
    predicted_price2 = np.exp(predicted_price)  # Converting from log scale
st.success(f"✅ Estimated Laptop Price: ₹{predicted_price2}")



 
