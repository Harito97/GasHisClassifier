# Create data version 2
import os
import shutil
import random
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Đường dẫn đến thư mục chứa dữ liệu gốc
data_dir = 'GasHisSDB/80/'
categories = ['Abnormal', 'Normal']

# Đường dẫn đến thư mục chứa dữ liệu mới
data_v2_dir = 'Data/datav2/'

# Tạo các thư mục train, valid, test
os.makedirs(data_v2_dir, exist_ok=True)
os.makedirs(os.path.join(data_v2_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(data_v2_dir, 'valid'), exist_ok=True)
os.makedirs(os.path.join(data_v2_dir, 'test'), exist_ok=True)

# Tạo các thư mục con cho mỗi loại
for category in categories:
    os.makedirs(os.path.join(data_v2_dir, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(data_v2_dir, 'valid', category), exist_ok=True)
    os.makedirs(os.path.join(data_v2_dir, 'test', category), exist_ok=True)

# Hàm để đọc ảnh và thực hiện phân cụm KMeans
def process_image(image_path):
    img = Image.open(image_path)
    img_np = np.array(img)
    w, h, d = img_np.shape
    img_reshaped = np.reshape(img_np, (w * h, d))
    
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(img_reshaped)
    labels = kmeans.predict(img_reshaped)
    clustered_img = np.zeros_like(img_np)
    
    for i in range(w):
        for j in range(h):
            clustered_img[i, j] = kmeans.cluster_centers_[labels[i * h + j]]
    
    clustered_img = np.clip(clustered_img, 0, 255)  # Đảm bảo các giá trị màu nằm trong khoảng [0, 255]
    return Image.fromarray(clustered_img.astype(np.uint8))

# Chia tập dữ liệu theo tỷ lệ 4:4:2
def split_data(image_paths):
    random.shuffle(image_paths)
    train_split = int(0.4 * len(image_paths))
    valid_split = int(0.8 * len(image_paths))
    
    train_paths = image_paths[:train_split]
    valid_paths = image_paths[train_split:valid_split]
    test_paths = image_paths[valid_split:]
    
    return train_paths, valid_paths, test_paths

# Di chuyển ảnh đến thư mục đích và xử lý phân cụm
def move_and_process_images(image_paths, destination_dir, category):
    for image_path in image_paths:
        img = process_image(image_path)
        dest_path = os.path.join(destination_dir, category, os.path.basename(image_path))
        img.save(dest_path)

# Thực hiện cho từng loại
for category in categories:
    image_paths = [os.path.join(data_dir, category, f) for f in os.listdir(os.path.join(data_dir, category)) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    train_paths, valid_paths, test_paths = split_data(image_paths)
    
    move_and_process_images(train_paths, os.path.join(data_v2_dir, 'train'), category)
    move_and_process_images(valid_paths, os.path.join(data_v2_dir, 'valid'), category)
    move_and_process_images(test_paths, os.path.join(data_v2_dir, 'test'), category)

print("Data processing and clustering completed.")
