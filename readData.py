import numpy as np

data = np.load("trainingData.npz")
theta = data["theta"]
d = data["d"]

print(theta)
print(d)

