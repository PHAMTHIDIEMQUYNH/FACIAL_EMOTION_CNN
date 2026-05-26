import os
import cv2
import numpy as np
import pickle
import random
from collections import Counter
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt

# ==================== CẤU HÌNH ====================
DATASET_PATHS = {
    'fer2013': 'data/fer2013',
    'ckplus':  'data/ckplus',
    'jaffe':   'data/jaffe',
    'sfew':    'data/sfew',
}

TARGET_SIZE = (48, 48)
USE_CLAHE   = True   # Bật CLAHE để cải thiện độ tương phản
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ==================== EMOTION MAPPING ====================
UNIFIED_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_TO_ID    = {e: i for i, e in enumerate(UNIFIED_EMOTIONS)}
EMOTION_NAMES    = UNIFIED_EMOTIONS

FER2013_MAP = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'sad',   5: 'surprise', 6: 'neutral',
}

# CK+ label 3 = contempt (không có trong 7 nhãn chuẩn).
# Mapping sang 'disgust' vì gần nghĩa nhất — có ghi chú rõ ràng.
CKPLUS_LABEL_MAP = {
    0: 'neutral',
    1: 'angry',
    2: 'disgust',
    3: 'disgust',   # contempt → disgust (closest proxy)
    4: 'fear',
    5: 'happy',
    6: 'sad',
    7: 'surprise',
}

JAFFE_MAP = {
    'AN': 'angry', 'DI': 'disgust', 'FE': 'fear',
    'HA': 'happy', 'SA': 'sad',     'SU': 'surprise', 'NE': 'neutral',
}

SFEW_MAP = {
    'Angry': 'angry', 'Disgust': 'disgust', 'Fear': 'fear',
    'Happy': 'happy', 'Sad': 'sad',         'Surprise': 'surprise',
    'Neutral': 'neutral',
}

# ==================== CLAHE ====================
def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Áp dụng CLAHE để tăng độ tương phản ảnh xám."""
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    return img


def _read_gray(path: str) -> np.ndarray | None:
    """Đọc ảnh, chuyển về xám, resize, áp CLAHE. Trả None nếu lỗi."""
    img = cv2.imread(path)
    if img is None:
        print(f"    ⚠️  Không đọc được: {path}")
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, TARGET_SIZE)
    img = apply_clahe(img)
    return img


# ==================== LOAD FER2013 ====================
# FER2013 đã có split sẵn → giữ nguyên cấu trúc train/test.
# Toàn bộ ảnh được load, split train/test sẽ tôn trọng nguồn gốc split này
# bằng cách đánh dấu qua trường `split` trong metadata.

def load_fer2013(base_path: str):
    """
    Trả về (images, labels, subjects, splits, sources)
    - subjects: None vì FER2013 không có subject ID
    - splits  : 'train' hoặc 'test' theo cấu trúc thư mục gốc
    """
    print(f"\n📁 Loading FER2013 from {base_path}")
    images, labels, subjects, splits, sources = [], [], [], [], []
    skipped = 0

    for split in ['train', 'test']:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            continue
        for emotion_folder in os.listdir(split_path):
            emotion_path = os.path.join(split_path, emotion_folder)
            if not os.path.isdir(emotion_path):
                continue
            emotion_name = emotion_folder.lower()
            if emotion_name not in EMOTION_TO_ID:
                continue
            emotion_id = EMOTION_TO_ID[emotion_name]
            for img_file in os.listdir(emotion_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img = _read_gray(os.path.join(emotion_path, img_file))
                if img is None:
                    skipped += 1
                    continue
                images.append(img)
                labels.append(emotion_id)
                subjects.append(None)   # không có subject ID
                splits.append(split)    # giữ nguyên split gốc
                sources.append('FER2013')

    print(f"  ✓ FER2013: {len(images)} ảnh  |  bỏ qua: {skipped}")
    return images, labels, subjects, splits, sources


# ==================== LOAD CK+ ====================
# CK+ chỉ dùng train folder; subject ID được lưu lại để
# thực hiện subject-aware split (tránh subject leakage).

def load_ckplus(base_path: str):
    print(f"\n📁 Loading CK+ from {base_path}")
    images, labels, subjects, splits, sources = [], [], [], [], []
    skipped = 0

    dataset_path = os.path.join(base_path, 'DATASET')
    label_csv    = os.path.join(base_path, 'train_labels.csv')

    if not os.path.exists(dataset_path):
        print("  ⚠️  Không tìm thấy thư mục DATASET")
        return images, labels, subjects, splits, sources
    if not os.path.exists(label_csv):
        print("  ⚠️  Không tìm thấy train_labels.csv")
        return images, labels, subjects, splits, sources

    df = pd.read_csv(label_csv)
    filename_to_label = dict(zip(df['image'], df['label']))
    print(f"  CSV: {len(df)} nhãn")

    # Chỉ dùng thư mục train — bỏ qua test để tránh leakage
    train_path = os.path.join(dataset_path, 'train')
    if not os.path.exists(train_path):
        print("  ⚠️  Không tìm thấy thư mục train")
        return images, labels, subjects, splits, sources

    for subject_id in os.listdir(train_path):
        subject_path = os.path.join(train_path, subject_id)
        if not os.path.isdir(subject_path):
            continue
        for img_file in os.listdir(subject_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if img_file not in filename_to_label:
                continue
            label_num = filename_to_label[img_file]
            if label_num not in CKPLUS_LABEL_MAP:
                continue
            emotion_name = CKPLUS_LABEL_MAP[label_num]
            if emotion_name not in EMOTION_TO_ID:
                continue
            img = _read_gray(os.path.join(subject_path, img_file))
            if img is None:
                skipped += 1
                continue
            images.append(img)
            labels.append(EMOTION_TO_ID[emotion_name])
            subjects.append(f"CKP_{subject_id}")   # subject ID để split
            splits.append('train')
            sources.append('CK+')

    print(f"  ✓ CK+: {len(images)} ảnh  |  bỏ qua: {skipped}")
    return images, labels, subjects, splits, sources


# ==================== LOAD JAFFE ====================
# JAFFE chỉ có 10 subject, ~213 ảnh → dùng để augment train,
# KHÔNG đưa vào test (tránh overestimate do tập quá nhỏ).

def load_jaffe(base_path: str):
    print(f"\n📁 Loading JAFFE from {base_path}")
    images, labels, subjects, splits, sources = [], [], [], [], []
    skipped = 0

    if not os.path.exists(base_path):
        print("  ⚠️  Không tìm thấy thư mục JAFFE")
        return images, labels, subjects, splits, sources

    for file in os.listdir(base_path):
        if not file.lower().endswith(('.tiff', '.tif')):
            continue
        parts = file.split('.')
        if len(parts) < 2:
            continue
        emotion_code = parts[1].upper()[:2]
        if emotion_code not in JAFFE_MAP:
            continue
        # Trích subject ID từ tên file (vd: KA.AN1.39 → subject = "KA")
        subject_id = parts[0][:2].upper()
        img = _read_gray(os.path.join(base_path, file))
        if img is None:
            skipped += 1
            continue
        images.append(img)
        labels.append(EMOTION_TO_ID[JAFFE_MAP[emotion_code]])
        subjects.append(f"JFF_{subject_id}")
        splits.append('train')   # JAFFE chỉ vào train pool
        sources.append('JAFFE')

    print(f"  ✓ JAFFE: {len(images)} ảnh  |  bỏ qua: {skipped}")
    print("  ℹ️  JAFFE chỉ dùng cho train pool (10 subjects, ~213 ảnh)")
    return images, labels, subjects, splits, sources


# ==================== LOAD SFEW ====================
# SFEW đã có benchmark split Train/Val/Test từ tác giả.
# → PHẢI tôn trọng split này, KHÔNG gộp rồi re-split ngẫu nhiên.

def load_sfew(base_path: str):
    print(f"\n📁 Loading SFEW from {base_path}")
    images, labels, subjects, splits, sources = [], [], [], [], []
    skipped = 0

    # Chú ý: chỉ map 'Train' và 'Val' sang 'train'; 'Test' giữ nguyên.
    # Val của SFEW → đưa vào validation pool riêng (không trộn vào train).
    sfew_split_map = {'Train': 'sfew_train', 'Val': 'sfew_val', 'Test': 'sfew_test'}

    for sfew_split, internal_split in sfew_split_map.items():
        split_path = os.path.join(base_path, sfew_split)
        if not os.path.exists(split_path):
            continue
        for emotion_folder in os.listdir(split_path):
            if emotion_folder not in SFEW_MAP:
                continue
            emotion_path = os.path.join(split_path, emotion_folder)
            if not os.path.isdir(emotion_path):
                continue
            emotion_id = EMOTION_TO_ID[SFEW_MAP[emotion_folder]]
            for img_file in os.listdir(emotion_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                img = _read_gray(os.path.join(emotion_path, img_file))
                if img is None:
                    skipped += 1
                    continue
                images.append(img)
                labels.append(emotion_id)
                subjects.append(None)
                splits.append(internal_split)
                sources.append('SFEW')

    print(f"  ✓ SFEW: {len(images)} ảnh  |  bỏ qua: {skipped}")
    print("  ℹ️  Giữ nguyên split Train/Val/Test benchmark của SFEW")
    return images, labels, subjects, splits, sources


# ==================== SUBJECT-AWARE SPLIT ====================

def subject_aware_test_split(images, labels, subjects, splits, sources,
                             test_size: float = 0.2):
    """
    Tách test set theo nguyên tắc:
    1. FER2013: dùng split gốc ('test') làm test pool, lấy 20% ngẫu nhiên
       từ đó nếu test_size < 1.0.
    2. SFEW   : dùng sfew_test làm test pool.
    3. CK+    : split theo subject ID (không để cùng 1 người ở train lẫn test).
    4. JAFFE  : hoàn toàn vào train — không đưa vào test.

    Trả về (train_idx, test_idx) — danh sách index trong mảng gốc.
    """
    images   = np.array(images,   dtype=object)
    labels   = np.array(labels)
    subjects = np.array(subjects, dtype=object)
    splits   = np.array(splits,   dtype=str)
    sources  = np.array(sources,  dtype=str)

    train_idx, test_idx = [], []

    # ---------- FER2013 ----------
    fer_mask = sources == 'FER2013'
    fer_train_mask = fer_mask & (splits == 'train')
    fer_test_mask  = fer_mask & (splits == 'test')
    train_idx.extend(np.where(fer_train_mask)[0].tolist())
    test_idx.extend(np.where(fer_test_mask)[0].tolist())

    # ---------- CK+ (subject-aware) ----------
    ckp_mask     = sources == 'CK+'
    ckp_indices  = np.where(ckp_mask)[0]
    ckp_subjects = subjects[ckp_indices]
    unique_subj  = np.unique([s for s in ckp_subjects if s is not None])

    if len(unique_subj) > 0:
        n_test_subj = max(1, int(len(unique_subj) * test_size))
        rng = np.random.default_rng(RANDOM_SEED)
        test_subjects = set(rng.choice(unique_subj, size=n_test_subj, replace=False))
        for idx in ckp_indices:
            if subjects[idx] in test_subjects:
                test_idx.append(int(idx))
            else:
                train_idx.append(int(idx))
        print(f"  CK+ subject split: "
              f"{len(unique_subj) - n_test_subj} train subjects, "
              f"{n_test_subj} test subjects")

    # ---------- JAFFE → chỉ train ----------
    jaffe_mask = sources == 'JAFFE'
    train_idx.extend(np.where(jaffe_mask)[0].tolist())

    # ---------- SFEW (giữ nguyên benchmark split) ----------
    sfew_train_mask = sources == 'SFEW'
    for idx in np.where(sfew_train_mask)[0]:
        sp = splits[idx]
        if sp == 'sfew_test':
            test_idx.append(int(idx))
        else:   # sfew_train + sfew_val → đưa vào train pool
            train_idx.append(int(idx))

    return np.array(train_idx), np.array(test_idx)


# ==================== MAIN LOADER ====================

def load_all_datasets(use_datasets=None):
    if use_datasets is None:
        use_datasets = ['fer2013', 'ckplus', 'jaffe', 'sfew']

    all_images, all_labels, all_subjects, all_splits, all_sources = [], [], [], [], []

    print("=" * 60)
    print("🔄 LOADING MULTIPLE DATASETS")
    print("=" * 60)

    loaders = {
        'fer2013': load_fer2013,
        'ckplus':  load_ckplus,
        'jaffe':   load_jaffe,
        'sfew':    load_sfew,
    }

    for name in use_datasets:
        path = DATASET_PATHS.get(name)
        if not path or not os.path.exists(path):
            print(f"⚠️  Dataset '{name}' không tìm thấy tại: {path}")
            continue
        imgs, lbls, subjs, spls, srcs = loaders[name](path)
        if imgs:
            all_images.extend(imgs)
            all_labels.extend(lbls)
            all_subjects.extend(subjs)
            all_splits.extend(spls)
            all_sources.extend(srcs)

    if not all_images:
        raise ValueError("❌ Không load được dữ liệu từ bất kỳ dataset nào!")

    print("\n" + "=" * 60)
    print("✅ TỔNG HỢP DỮ LIỆU")
    print("=" * 60)
    print(f"  Tổng: {len(all_images)} ảnh")

    label_counts = Counter(all_labels)
    for eid, cnt in sorted(label_counts.items()):
        pct = cnt / len(all_labels) * 100
        bar = '█' * int(pct / 2)
        print(f"  {EMOTION_NAMES[eid]:10s}: {cnt:5d} ({pct:5.1f}%) {bar}")

    src_counts = Counter(all_sources)
    print("\n  Theo nguồn:")
    for src, cnt in src_counts.items():
        print(f"  {src:10s}: {cnt:5d}")

    return all_images, all_labels, all_subjects, all_splits, all_sources


# ==================== TIỀN XỬ LÝ ====================

def preprocess_images(images) -> np.ndarray:
    arr = np.array(images, dtype='float32') / 255.0
    return np.expand_dims(arr, axis=-1)


def create_validation_set(X_train, y_train, val_size=0.1):
    print(f"\n📊 Tách validation ({val_size*100:.0f}% từ train)...")
    y_int = np.argmax(y_train, axis=1)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size, random_state=RANDOM_SEED, stratify=y_int
    )
    print(f"  Train: {len(X_tr)}  |  Val: {len(X_val)}")
    return X_tr, X_val, y_tr, y_val


def compute_class_weights(y_train_encoded) -> dict:
    y_int = np.argmax(y_train_encoded, axis=1)
    cw = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
    return dict(enumerate(cw))


def get_augmentation_config() -> dict:
    """Trả về config augmentation dưới dạng dict để lưu vào pkl."""
    return {
        'rotation_range':    15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'shear_range':       0.1,
        'zoom_range':        0.1,
        'horizontal_flip':   True,
        'brightness_range':  [0.8, 1.2],
        'fill_mode':         'nearest',
    }


def create_augmentation_generator() -> ImageDataGenerator:
    return ImageDataGenerator(**get_augmentation_config())


# ==================== SAVE ====================

def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test,
                            encoder, class_weights,
                            save_path='utils/preprocessed_data.pkl'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_int = np.argmax(y_train, axis=1)
    class_dist_named = {EMOTION_NAMES[k]: v
                        for k, v in Counter(y_int.tolist()).items()}

    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val':   X_val,
        'y_val':   y_val,
        'X_test':  X_test,
        'y_test':  y_test,
        'encoder': encoder,
        'class_weights': class_weights,
        'augmentation_config': get_augmentation_config(),   # ← lưu config augment
        'metadata': {
            'num_classes':        7,
            'image_shape':        X_train.shape[1:],
            'train_samples':      len(X_train),
            'val_samples':        len(X_val),
            'test_samples':       len(X_test),
            'emotion_names':      EMOTION_NAMES,
            'class_distribution': class_dist_named,
            'used_clahe':         USE_CLAHE,
            'random_seed':        RANDOM_SEED,
            'split_strategy':     (
                'FER2013: split gốc | '
                'CK+: subject-aware | '
                'JAFFE: train only | '
                'SFEW: benchmark split'
            ),
        },
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\n💾 Đã lưu → {save_path}")
    print(f"  Phân bố class (train): {class_dist_named}")


# ==================== VISUALIZE ====================

def visualize_samples(images, labels, num_samples=16, save_path='utils/data_samples.png'):
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    cols = 4
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(indices):
            img = images[indices[i]].squeeze()
            y = labels[indices[i]]
            label_idx = int(np.argmax(y)) if (np.ndim(y) > 0 and len(y) > 1) else int(y)
            ax.imshow(img, cmap='gray')
            ax.set_title(EMOTION_NAMES[label_idx], fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Lưu ảnh mẫu → {save_path}")


def visualize_split_distribution(y_train, y_val, y_test,
                                  save_path='utils/split_distribution.png'):
    """Vẽ biểu đồ phân bố nhãn qua các split để kiểm tra cân bằng."""
    splits = {'Train': y_train, 'Val': y_val, 'Test': y_test}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, y_enc) in zip(axes, splits.items()):
        y_int = np.argmax(y_enc, axis=1)
        counts = [Counter(y_int.tolist()).get(i, 0) for i in range(7)]
        ax.bar(EMOTION_NAMES, counts, color='steelblue', alpha=0.8)
        ax.set_title(f"{name} ({len(y_int)} ảnh)")
        ax.set_xticklabels(EMOTION_NAMES, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Số lượng')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Lưu biểu đồ phân bố → {save_path}")


# ==================== MAIN ====================

def main():
    print("\n" + "=" * 60)
    print("🚀 MULTI-DATASET PREPROCESSING")
    print("   FER2013 + CK+ + JAFFE + SFEW")
    print(f"   CLAHE: {'ON' if USE_CLAHE else 'OFF'} | Seed: {RANDOM_SEED}")
    print("=" * 60)

    # 1. Load toàn bộ dữ liệu với metadata đầy đủ
    all_images, all_labels, all_subjects, all_splits, all_sources = load_all_datasets()

    # 2. Tách train/test theo chiến lược phù hợp từng dataset
    print("\n📂 Tách train/test (subject-aware + benchmark splits)...")
    train_idx, test_idx = subject_aware_test_split(
        all_images, all_labels, all_subjects, all_splits, all_sources,
        test_size=0.2
    )

    # Shuffle train index (đã seed)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(train_idx)

    all_images_arr = np.array(all_images, dtype=object)
    all_labels_arr = np.array(all_labels)

    X_train_raw = np.stack(all_images_arr[train_idx])
    y_train_raw = all_labels_arr[train_idx]
    X_test_raw  = np.stack(all_images_arr[test_idx])
    y_test_raw  = all_labels_arr[test_idx]

    print(f"\n  Train pool : {len(X_train_raw)} ảnh")
    print(f"  Test  pool : {len(X_test_raw)} ảnh")

    # 3. Tiền xử lý ảnh
    X_train_proc = preprocess_images(X_train_raw)
    X_test_proc  = preprocess_images(X_test_raw)

    # 4. Encode nhãn
    encoder = LabelBinarizer()
    y_train_enc = encoder.fit_transform(y_train_raw)
    y_test_enc  = encoder.transform(y_test_raw)

    # 5. Tách validation từ train
    X_train_final, X_val, y_train_final, y_val = create_validation_set(
        X_train_proc, y_train_enc, val_size=0.1
    )

    # 6. Class weights
    class_weights = compute_class_weights(y_train_final)
    print(f"\n⚖️  Class weights: {class_weights}")

    # 7. Augmentation generator (dùng ngay khi training)
    datagen = create_augmentation_generator()

    # 8. Lưu
    save_preprocessed_data(
        X_train_final, y_train_final,
        X_val, y_val,
        X_test_proc, y_test_enc,
        encoder, class_weights,
    )

    # 9. Visualize
    visualize_samples(X_train_final, y_train_final)
    visualize_split_distribution(y_train_final, y_val, y_test_enc)

    print("\n" + "=" * 60)
    print("✨ PREPROCESSING HOÀN TẤT!")
    print("=" * 60)

    return {
        'X_train':       X_train_final,
        'y_train':       y_train_final,
        'X_val':         X_val,
        'y_val':         y_val,
        'X_test':        X_test_proc,
        'y_test':        y_test_enc,
        'encoder':       encoder,
        'class_weights': class_weights,
        'datagen':       datagen,
    }


if __name__ == '__main__':
    data = main()
    print("\n✅ Sẵn sàng training! Dùng file utils/preprocessed_data.pkl")