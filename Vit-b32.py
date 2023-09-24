import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from torch.utils.data import Subset


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)


def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


lessDates=True  # lo uso per  usare il  modello  di  training  ridotto a  100 immagini ma non da precisione/Accuracy

if (lessDates):
    # Define the indices for the subset. Here, we take the first 50 for training and the next 50 for testing
    train_indices = range(0, 70)
    test_indices = range(70, 100)

    # Create the subset datasets
    train_subset = Subset(train, train_indices)
    test_subset = Subset(test, test_indices)

    # Now you can continue with your code but use train_subset and test_subset instead of train and test
    train_features, train_labels = get_features(train_subset)
    test_features, test_labels = get_features(test_subset)
else:
    # Calculate the image features
    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)


# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
predictions_train=classifier.fit(train_features, train_labels).predict(train_features)

accuracy=classifier.score(test_features,test_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
# accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
train_accuracy=np.mean((train_labels == predictions_train).astype(float)) * 100
print(f"Score = {accuracy*100:.3f}  Score_data_Train:  {train_accuracy:.3f} ")


from sklearn.metrics import f1_score  # Import the F1 score metric
# Evaluate using the logistic regression classifier
#predictions = classifier.predict(test_features)

# Calculate Accuracy
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

# Calculate F1 Score
f1 = f1_score(test_labels, predictions, average='weighted')  # 'weighted' takes into account label imbalance
print(f"F1 Score = {f1:.3f}")

from  sklearn.metrics import  classification_report
report=classification_report(test_labels,predictions)

print (report)