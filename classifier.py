import numpy as np
import pandas
import os
import cv2
import matplotlib.pyplot as plt
import dexpy

# Load data in Testing/glioma_tumor directory into a dataframe, downsizing the images to 50x50

cancerous_dirs = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
healthy_dir = 'no_tumor'

def load_data(dir):
    data = []
    for root, dirs, files in os.walk(f"Testing/{dir}"):
        for file in files:
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (100, 100))
                data.append(img.flatten())
    
    # Return the data as a numpy array 
    return np.array(data)

glioma_pics = load_data(cancerous_dirs[0])
meningioma_pics = load_data(cancerous_dirs[1])
pituitary_pics = load_data(cancerous_dirs[2])
healthy_pics = load_data(healthy_dir)
cancerous_pics = np.concatenate((glioma_pics, meningioma_pics, pituitary_pics))

# Create a dataframe with the data, with images in one column and labels in another
df = pandas.DataFrame(np.concatenate((cancerous_pics, healthy_pics)))
df['label'] = np.concatenate((np.ones(len(cancerous_pics)), np.zeros(len(healthy_pics))))

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

from sklearn.model_selection import train_test_split

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Perform a logistic regression on the data using only numpy
def sigmoid(z):
    return 1 / (1 + np.exp(-z))




