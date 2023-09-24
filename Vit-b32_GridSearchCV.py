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


lessDates=False  # lo uso per  usare il  modello  di  training  ridotto a  100 immagini ma non da precisione/Accuracy

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
    #SECTION:   serializzo il dataset  numpy con gli embeddings  

    import os

    if os.path.exists('train_features.npy') and os.path.exists('train_labels.npy'):
        train_features = np.load('train_features.npy')
        train_labels = np.load('train_labels.npy')
    else:
        train_features, train_labels = get_features(train)
        np.save('train_features.npy', train_features)
        np.save('train_labels.npy', train_labels)

    # train_features, train_labels = get_features(train)
    if os.path.exists('test_features.npy') and os.path.exists('test_labels.npy'):
        test_features = np.load('test_features.npy')
        test_labels = np.load('test_labels.npy')
    else:
        test_features, test_labels = get_features(test)
        np.save('test_features.npy', test_features)
        np.save('test_labels.npy', test_labels)
    
    # test_features, test_labels = get_features(test)

#SECTION:   GridSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Definizione della griglia di iperparametri
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'max_iter': [100, 500, 1000]}

# GridSearchCV
grid = GridSearchCV(LogisticRegression(), param_grid, scoring='accuracy', cv=5)
grid.fit(train_features, train_labels)

# Migliori iperparametri trovati
best_params = grid.best_params_
print(f"Best parameters: {best_params}")

# # Addestramento del modello con i migliori iperparametri
# best_model = LogisticRegression(**best_params)
# best_model.fit(X_train, y_train)

# # Valutazione del modello
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")



# Perform logistic regression
classifier = LogisticRegression(**best_params, verbose=1)
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