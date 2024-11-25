import numpy as np 

data = np.load("data/preprocessed_data/nino_dataset_1m.npz")


# Accéder aux variables stockées
X = data["X"]
y = data["y"]

# Vérifier les dimensions
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("First 5 elements of y:", y[:20])