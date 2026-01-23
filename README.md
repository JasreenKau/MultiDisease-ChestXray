# 🩻 Multi-Disease Diagnosis from Chest X-Rays

This project uses **deep learning** to perform **multi-label classification** of chest diseases from X-ray images.  
It is based on the **NIH ChestX-ray14 dataset** and includes modules for **training, evaluation,** and **model explainability**.

## 📌 Features

- Multi-label classification of **14 chest diseases**.
- End-to-end pipeline: **Training, evaluation,** and **prediction**.
- Model explainability and visualization support,
- Detailed performance metrics:
  - Classification report
  - ROC curves
  - Prediction analysis

## 📂 Project Structure

```text
Multi-Disease-Diagnosis-from-Chest-X-Rays/
│
├── main.py                   # Entry point for training & evaluation
├── chest_xray.ipynb          # Jupyter notebook workflow
├── resize_images.py          # Image preprocessing script
├── requirements.txt          # Project dependencies
│
├── checkpoints/              # Saved trained models
├── evaluation_results/       # Metrics, plots, and predictions
│
├── src/
│   ├── dataloader.py         # Data loading and preprocessing
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation and metrics
│   └── model.py              # Model architectures
│
└── README.md                 # Project documentation
```

## 📊 Dataset
The project uses the NIH ChestX-ray14 dataset:
- **Source:** [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Images:** 112,120 frontal-view X-rays from 30,805 patients.
- **Labels:** 14 disease categories (multi-label).
- **Note:** Dataset is not included in this repository due to size limits.

## ⚙️ Installation
1. Clone the repository
```bash
git clone https://github.com/JasreenKaur/ChestXray-MultiDisease.git
cd MultiDisease-ChestXray
```
2. Create and activate a virtual environment
```bash
conda create -n chestxray python=3.9
conda activate chestxray
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Usage
1. Train the model
```bash
python main.py --mode train --epochs 20 --batch_size 32
```
2. Evaluate the model
```bash
python main.py --mode evaluate
```
3. Predict from an image
```bash
python main.py --mode predict --input_path path/to/image.jpg
```

## 📥 Dataset Preparation
1. Download the dataset from the [NIH ChestX-ray14 dataset page](https://nihcc.app.box.com/v/ChestXray-NIHCC).
2. Extract images into `data/`.
3. Run preprocessing:
```bash
python resize_images.py --input_dir data/images --output_dir data/resized
```

## 📌 Notes
- GPU support (TensorFlow + CUDA) is highly recommended for faster training.
- Large files are excluded using `.gitignore`.
- Suitable for research, academic projects, and experimentation.

## 📜 License
This project is licensed under the **MIT License**.
See the [LICENSE](https://github.com/JasreenKau/MultiDisease-ChestXray/blob/main/LICENSE) file for more details.

## 👤 Author
**Jasreen** [GitHub Profile](https://github.com/JasreenKau)
