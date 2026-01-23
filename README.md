# Multi-Disease Diagnosis from Chest X-rays

This project uses deep learning to perform multi-label classification of chest diseases from X-ray images.
It is based on the NIH ChestX-ray14 dataset and includes training,and evaluation explainability.

## 📌 Features
- Multi-label classification of 14 chest diseases.
- Training, evaluation, and prediction scripts.
- Visualizations for model explainability.
- Metrics reporting (classification report, ROC curves, etc.).

## 📂 Project Structure
Multi-Disease-Diagnosis-from-Chest-X-rays/
├── main.py                  #Entry point for training & evaluation
├── chest\_xray.ipynb        #Jupyter Notebook workflow
├── Checkpoints              #Saved Model
├── src/
│   ├── dataloader.py        #Data loading and preprocessing
│   ├── train.py             #Training logic
│   ├── evaluate.py          #Evaluation and metrics
│   ├── model.py             #Model architectures
│   ├── requirements.txt     
├── evaluation               #Saved metrics and predictions
└── README.md                #Project documentation

## 📊 Dataset
The project uses the NIH ChestX-ray14 dataset:
- Source: NIH Clinical Center
- Images: 112,120 frontal-view X-rays from 30,805 patients.
- Labels: 14 disease categories (multi-label).
- Note: Dataset is not included in this repository due to size limits.

## ⚙️ Installation
- Clone the repository
git clone https://github.com/kimirandhawa/ChestXray-MultiDisease.git
cd ChestXray-MultiDisease
- Create and activate a virtual environment
conda create -n chestxray python=3.9
conda activate chestxray
- Install dependencies
pip install -r requirements.txt

## ▶️ Usage
1. Train the model
python main.py --mode train --epochs 20 --batch_size 32
2. Evaluate the model
python main.py --mode evaluate
3. Predict from an image
python main.py --mode predict --input_path path/to/image.jpg

## 📥 Dataset Preparation
- Download from the NIH ChestX-ray14 dataset page.
- Extract images into data/.
- Run preprocessing:
python resize_images.py --input_dir data/images --output_dir data/resized

## 📌 Notes
- Ensure TensorFlow GPU support for faster training.
- Large files are excluded via .gitignore.

##📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.

## 👤 Author
Jasreen GitHub Profile
