import os
import sys

#suppress OpenCV warnings
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)

from src.model import create_model
from src.train import train_model
from src.dataloader import get_data_generators
from src.evaluate import evaluate_model
from src.visualize import plot_metrics

#load data
train_gen, val_gen, test_gen = get_data_generators(
    csv_path="/content/drive/MyDrive/multidisease detection/data/Data_Entry_2017.csv",
    image_dir="/content/drive/MyDrive/multidisease detection/data/images_resized",
    batch_size=64
)



#Create model
model = create_model()

#Train and save model 
history = train_model(model, train_gen, val_gen, epochs=5)

#Plot training metrics and save
plot_metrics(history)

#Evaluate model
evaluate_model(model, test_gen)
