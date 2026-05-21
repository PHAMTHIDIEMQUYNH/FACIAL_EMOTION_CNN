import os
import cv2
import numpy as np
import pickle
import random
from collections import Counter

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Cách import đúng cho TensorFlow
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

import matplotlib.pyplot as plt
# CẤU HÌNH
DATASET_PATH = 'data'          # Thư mục gốc chứa train và test
TARGET_SIZE = (48, 48)         # Kích thước ảnh đầu ra (48x48)
BATCH_SIZE = 64                # Batch size cho training

# Mapping emotion
EMOTION_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# LOAD DỮ LIỆU 

def load_images_from_folder(folder_path):
   
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    if not os.path.exists(folder_path):
        print(f" Thư mục không tồn tại: {folder_path}")
        return images
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            try:
                # Đọc ảnh dưới dạng grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize về kích thước chuẩn (48,48)
                    img = cv2.resize(img, TARGET_SIZE)
                    images.append(img)
                else:
                    print(f" Không thể đọc ảnh: {img_path}")
            except Exception as e:
                print(f"Lỗi khi đọc {img_path}: {e}")
    
    return images

def load_dataset_from_structure(base_path=DATASET_PATH):

    print("="*60)
    print("BẮT ĐẦU LOAD DỮ LIỆU TỪ THƯ MỤC")
    print("="*60)
    
    # Kiểm tra thư mục gốc
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục {base_path}")
    
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_path}")
    
    # Load train data 
    print(f"\n Đang load TRAIN data từ: {train_path}")
    X_train, y_train = [], []
    
    for emotion_name, emotion_id in EMOTION_MAP.items():
        emotion_folder = os.path.join(train_path, emotion_name)
        if os.path.exists(emotion_folder):
            images = load_images_from_folder(emotion_folder)
            print(f"  - {emotion_name}: {len(images)} ảnh")
            X_train.extend(images)
            y_train.extend([emotion_id] * len(images))
        else:
            print(f" Không tìm thấy: {emotion_folder}")
    
    # --- Load test data ---
    print(f"\n Đang load TEST data từ: {test_path}")
    X_test, y_test = [], []
    
    for emotion_name, emotion_id in EMOTION_MAP.items():
        emotion_folder = os.path.join(test_path, emotion_name)
        if os.path.exists(emotion_folder):
            images = load_images_from_folder(emotion_folder)
            print(f"  - {emotion_name}: {len(images)} ảnh")
            X_test.extend(images)
            y_test.extend([emotion_id] * len(images))
        else:
            print(f" Không tìm thấy: {emotion_folder}")
    
    # Chuyển đổi sang numpy array
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)
    
    # Kiểm tra dữ liệu rỗng
    if len(X_train) == 0:
        raise ValueError(" Không có ảnh nào trong thư mục train!")
    if len(X_test) == 0:
        raise ValueError(" Không có ảnh nào trong thư mục test!")
    
    print(f"\n LOAD DỮ LIỆU THÀNH CÔNG!")
    print(f"  - Train: {X_train.shape[0]} ảnh")
    print(f"  - Test: {X_test.shape[0]} ảnh")
    print(f"  - Kích thước ảnh: {X_train.shape[1]}x{X_train.shape[2]}")
    print(f"  - Giá trị pixel: [{X_train.min():.0f}, {X_train.max():.0f}]")
    
    return X_train, y_train, X_test, y_test


# TIỀN XỬ LÝ ẢNH 

def preprocess_images(images):
   
    print("\n Đang tiền xử lý ảnh...")
    
    # Normalization (0-255 -> 0-1)
    images_normalized = images.astype('float32') / 255.0
    
    # Thêm channel dimension cho CNN
    images_processed = np.expand_dims(images_normalized, axis=-1)
    
    print(f" Đã normalize ảnh về [0, 1]")
    print(f" Shape sau xử lý: {images_processed.shape}")
    
    return images_processed


# ENCODE LABELS 

def encode_labels(y_train, y_test):
    
    print("\n Đang xử lý nhãn...")
    
    # Thống kê phân bố nhãn
    print("Phân bố nhãn TRAIN:")
    train_counts = Counter(y_train)
    for emotion_id, count in sorted(train_counts.items()):
        print(f"  {EMOTION_NAMES[emotion_id]}: {count} ảnh ({count/len(y_train)*100:.1f}%)")
    
    print("\nPhân bố nhãn TEST:")
    test_counts = Counter(y_test)
    for emotion_id, count in sorted(test_counts.items()):
        print(f"  {EMOTION_NAMES[emotion_id]}: {count} ảnh ({count/len(y_test)*100:.1f}%)")
    
    # One-hot encoding
    encoder = LabelBinarizer()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    print(f"\n Đã encode labels (shape: {y_train_encoded.shape})")
    
    return y_train_encoded, y_test_encoded, encoder


# TẠO VALIDATION SET

def create_validation_set(X_train, y_train, val_size=0.1, random_state=42):

    print(f"\n Đang tách validation set ({val_size*100:.0f}% từ train)...")
    
    # Lấy nhãn dạng số để phân tầng
    y_train_int = np.argmax(y_train, axis=1)
    
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_int
    )
    
    print(f"✓ Train mới: {X_train_new.shape[0]} ảnh")
    print(f"✓ Validation: {X_val.shape[0]} ảnh")
    
    return X_train_new, X_val, y_train_new, y_val


# DATA AUGMENTATION 

def create_augmentation_generator():
    
    print("\n Cấu hình data augmentation:")
    
    datagen = ImageDataGenerator(
        rotation_range=20,          # Xoay ±20 độ
        width_shift_range=0.1,      # Dịch ngang 10%
        height_shift_range=0.1,     # Dịch dọc 10%
        zoom_range=0.1,             # Zoom ±10%
        horizontal_flip=True,       # Lật ngang
        fill_mode='nearest'         # Điền pixel gần nhất
    )
    
    print("  ✓ Rotation: ±20°")
    print("  ✓ Width shift: ±10%")
    print("  ✓ Height shift: ±10%")
    print("  ✓ Zoom: ±10%")
    print("  ✓ Horizontal flip: True")
    
    return datagen


# LƯU DỮ LIỆU 

def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test, 
                          encoder, save_path='utils/preprocessed_data.pkl'):
    
    print(f"\n Đang lưu dữ liệu vào {save_path}...")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Đóng gói dữ liệu
    preprocessed_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'encoder': encoder,
        'metadata': {
            'num_classes': 7,
            'image_shape': X_train.shape[1:],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'emotion_names': EMOTION_NAMES
        }
    }
    
    # Ghi file
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    file_size = os.path.getsize(save_path) / 1024 / 1024
    print(f"✓ Đã lưu thành công! (Kích thước: {file_size:.2f} MB)")


# TRỰC QUAN HÓA DỮ LIỆU 

def visualize_samples(images, labels, num_samples=10, save_path='utils/data_samples.png'):
   
    # Chọn ngẫu nhiên num_samples ảnh
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    # Tạo figure
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(indices):
            i = indices[idx]
            # Lấy ảnh (bỏ channel dimension)
            img = images[i].squeeze()
            
            # Lấy nhãn - SỬA PHẦN NÀY
            label_data = labels[i]
            # Kiểm tra và chuyển đổi label về số nguyên
            if hasattr(label_data, 'shape') and len(label_data.shape) > 1:
                label_idx = np.argmax(label_data)  # one-hot
            elif hasattr(label_data, '__len__') and len(label_data) > 1:
                label_idx = np.argmax(label_data)  # array-like
            else:
                label_idx = int(label_data) if hasattr(label_data, '__int__') else label_data
            
            # Đảm bảo là số nguyên
            label_idx = int(label_idx)
            emotion = EMOTION_NAMES[label_idx]
            
            ax.imshow(img, cmap='gray')
            ax.set_title(emotion, fontsize=12, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle('FER Dataset Samples (48x48 grayscale)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Lưu ảnh
    os.makedirs('utils', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Đã lưu ảnh mẫu vào: {save_path}")
    plt.close()


# MAIN PIPELINE

def main():
    
    print("\n" + "="*60)
    print(" BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU CẢM XÚC KHUÔN MẶT")
    print("="*60)
    
    # 1. Load dữ liệu thô từ thư mục
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_dataset_from_structure(DATASET_PATH)
    
    # 2. Tiền xử lý ảnh (normalize + channel)
    X_train_processed = preprocess_images(X_train_raw)
    X_test_processed = preprocess_images(X_test_raw)
    
    # 3. Encode labels (one-hot)
    y_train_encoded, y_test_encoded, encoder = encode_labels(y_train_raw, y_test_raw)
    
    # 4. Tạo validation set (10% từ train)
    X_train_final, X_val, y_train_final, y_val = create_validation_set(
        X_train_processed, y_train_encoded, val_size=0.1
    )
    
    # 5. Tạo data augmentation generator (cho team train sử dụng)
    datagen = create_augmentation_generator()
    
    # 6. Lưu dữ liệu đã xử lý
    save_preprocessed_data(
        X_train_final, y_train_final,
        X_val, y_val,
        X_test_processed, y_test_encoded,
        encoder
    )
    
    # 7. Trực quan hóa dữ liệu
    visualize_samples(X_train_final, y_train_final)
    
    # 8. In kết quả
    print("\n" + "="*60)
    print(" TIỀN XỬ LÝ DỮ LIỆU HOÀN TẤT!")
    print("="*60)
    print("\n OUTPUT:")
    print("   utils/preprocessed_data.pkl - Dữ liệu đã xử lý")
    print("   utils/data_samples.png - Ảnh mẫu kiểm tra")
    
    return {
        'X_train': X_train_final,
        'y_train': y_train_final,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test_processed,
        'y_test': y_test_encoded,
        'encoder': encoder
    }


# HÀM TIỆN ÍCH CHO MEMBER 

def load_preprocessed_data():
   
    with open('utils/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("\n Đã load dữ liệu đã tiền xử lý:")
    print(f"  - Train: {data['X_train'].shape}")
    print(f"  - Validation: {data['X_val'].shape}")
    print(f"  - Test: {data['X_test'].shape}")
    
    return data


# CHẠY CHÍNH 
if __name__ == '__main__':
    # Chạy pipeline
    data = main()
    
    # Kiểm tra nhanh dữ liệu
    print("\n KIỂM TRA DỮ LIỆU:")
    print(f"  - X_train dtype: {data['X_train'].dtype}")
    print(f"  - X_train range: [{data['X_train'].min():.3f}, {data['X_train'].max():.3f}]")
    print(f"  - y_train shape: {data['y_train'].shape}")
    print(f"  - Sample label: {EMOTION_NAMES[np.argmax(data['y_train'][0])]}")

