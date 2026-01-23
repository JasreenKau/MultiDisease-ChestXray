import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import warnings
import logging

#suppress OpenCV and TF warnings
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

#Preprocessing helpers
def preprocess_image(img):
    """Apply CLAHE, resize, normalize, and expand dims for grayscale."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

def multilabel_binarize(y_batch):
    """Convert list of label lists into binary array."""
    binarized = np.zeros((len(y_batch), len(ALL_LABELS)), dtype=np.float32)
    for i, labels in enumerate(y_batch):
        for label in labels:
            if label in ALL_LABELS:
                binarized[i, ALL_LABELS.index(label)] = 1
    return binarized

def clean_metadata(df):
    """Remove invalid rows and duplicates."""
    df = df[df['Finding Labels'].notna()]
    df = df.drop_duplicates(subset='Image Index')
    df['Finding Labels'] = df['Finding Labels'].replace('No Finding', '')
    return df

#Custom Generator
class ChestXrayGenerator(Sequence):
    def __init__(self, df, image_dir, batch_size=32, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        X, y_labels = [], []
        for _, row in batch_df.iterrows():
            img_path = os.path.join(self.image_dir, row['Image Index'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                with open("corrupt_images.txt", "a") as f:
                    f.write(img_path + "\n")
                continue
            X.append(preprocess_image(img))
            y_labels.append(row['Finding Labels'].split('|'))

        X = np.array(X, dtype=np.float32)
        y = multilabel_binarize(y_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

#Loader function
def get_data_generators(csv_path, image_dir, batch_size=32):
    df = pd.read_csv(csv_path)
    df = clean_metadata(df)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

    train_gen = ChestXrayGenerator(train_df, image_dir=image_dir, batch_size=batch_size)
    val_gen = ChestXrayGenerator(val_df, image_dir=image_dir, batch_size=batch_size)
    test_gen = ChestXrayGenerator(test_df, image_dir=image_dir, batch_size=batch_size, shuffle=False)

    return train_gen, val_gen, test_gen
