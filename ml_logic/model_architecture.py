#Thiết kế khung mạng CNN

import tensorflow as tf
from tensorflow.keras import layers, models

def create_emotion_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Hàm khởi tạo cấu trúc mạng CNN nhận diện cảm xúc khuôn mặt sử dụng Keras Sequential API.
    
    Tham số:
    - input_shape: tuple, kích thước ma trận ảnh đầu vào (mặc định 48x48x1).
    - num_classes: int, số lượng phân lớp cảm xúc đầu ra (mặc định 7).
    """
    model = models.Sequential(name="Facial_Emotion_CNN")
    
    # Sử dụng bộ khởi tạo trọng số He Normal phù hợp nhất với hàm kích hoạt ReLU
    initializer = 'he_normal'
    
    # =========================================================================
    # KHỐI TÍCH CHẬP 1 (32 Filters) - Trích xuất đặc trưng hình học cơ bản
    # Kích thước đặc trưng: (48, 48, 1) -> (48, 48, 32) -> (24, 24, 32)
    # =========================================================================
    model.add(layers.Conv2D(
        filters=32, kernel_size=(3, 3), padding='same', 
        activation='relu', kernel_initializer=initializer,
        input_shape=input_shape, name="conv_1_1"
    ))
    model.add(layers.BatchNormalization(name="batch_norm_1_1"))
    
    model.add(layers.Conv2D(
        filters=32, kernel_size=(3, 3), padding='same', 
        activation='relu', kernel_initializer=initializer, 
        name="conv_1_2"
    ))
    model.add(layers.BatchNormalization(name="batch_norm_1_2"))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(layers.Dropout(0.25, name="dropout_1"))
    
    # =========================================================================
    # KHỐI TÍCH CHẬP 2 (64 Filters) - Trích xuất đặc trưng trung cấp (Mắt, Mũi, Miệng)
    # Kích thước đặc trưng: (24, 24, 32) -> (24, 24, 64) -> (12, 12, 64)
    # =========================================================================
    model.add(layers.Conv2D(
        filters=64, kernel_size=(3, 3), padding='same', 
        activation='relu', kernel_initializer=initializer, 
        name="conv_2_1"
    ))
    model.add(layers.BatchNormalization(name="batch_norm_2_1"))
    
    model.add(layers.Conv2D(
        filters=64, kernel_size=(3, 3), padding='same', 
        activation='relu', kernel_initializer=initializer, 
        name="conv_2_2"
    ))
    model.add(layers.BatchNormalization(name="batch_norm_2_2"))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"))
    model.add(layers.Dropout(0.25, name="dropout_2"))
    
    # =========================================================================
    # KHỐI TÍCH CHẬP 3 (128 Filters) - Trích xuất cấu trúc tổng thể tổ hợp cơ mặt
    # Kích thước đặc trưng: (12, 12, 64) -> (12, 12, 128) -> (6, 6, 128)
    # =========================================================================
    model.add(layers.Conv2D(
        filters=128, kernel_size=(3, 3), padding='same', 
        activation='relu', kernel_initializer=initializer, 
        name="conv_3_1"
    ))
    model.add(layers.BatchNormalization(name="batch_norm_3_1"))
    
    model.add(layers.Conv2D(
        filters=128, kernel_size=(3, 3), padding='same', 
        activation='relu', kernel_initializer=initializer, 
        name="conv_3_2"
    ))
    model.add(layers.BatchNormalization(name="batch_norm_3_2"))
    
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_3"))
    model.add(layers.Dropout(0.25, name="dropout_3"))

    # =========================================================================
    # KHỐI PHÂN LOẠI (Fully Connected Layer) - Ra quyết định xác suất cảm xúc
    # Duỗi phẳng ma trận đặc trưng từ 6x6x128 thành vector phẳng 1D (4.608 phần tử)
    # =========================================================================
    model.add(layers.Flatten(name="flatten"))
    
    model.add(layers.Dense(
        units=256, activation='relu', 
        kernel_initializer=initializer, name="dense_hidden"
    ))
    model.add(layers.BatchNormalization(name="batch_norm_dense"))
    model.add(layers.Dropout(0.5, name="dropout_dense"))
    
    # Tầng đầu ra sử dụng Softmax để lấy phân phối xác suất của 7 lớp cảm xúc
    model.add(layers.Dense(
        units=num_classes, activation='softmax', name="output_layer"
    ))
    
    return model

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 ĐANG KHỞI TẠO VÀ KIỂM TRA KIẾN TRÚC MÔ HÌNH CNN DÀNH CHO FER...")
    print("="*60)
    
    # Khởi tạo mô hình
    model = create_emotion_cnn()
    
    # In bảng tóm tắt kiến trúc mạng
    model.summary()
    
    print("\n" + "="*60)
    print("✅ KIỂM TRA THÀNH CÔNG: Cấu trúc tầng hợp lệ, sẵn sàng huấn luyện!")
    print("="*60)
