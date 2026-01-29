from modelClasses import CNF, trainingDataSet

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

EPSILON = 1e-6

dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
dataPoisson = torch.tensor(np.load("/raid/vigneshk/data/poissonData.npy")).float()
thetaData = torch.tensor(np.load("/raid/vigneshk/data/theta_data.npy")).float()
thetaStandard = torch.tensor(np.load("/raid/vigneshk/data/thetaData_standard.npy")).float()


latent_mean = dataPoisson_latent.mean(dim=0)
latent_stDev = dataPoisson_latent.std(dim=0, correction=0)

dataPoisson_latent_Standard = (dataPoisson_latent - latent_mean) / (latent_stDev + EPSILON)

dataMap = trainingDataSet(thetaStandard,dataPoisson_latent_Standard)

dL = DataLoader(dataMap, batch_size = 1024, shuffle = True)


n_features = int(thetaData.shape[1])
context_features = int(dataPoisson_latent_Standard.shape[1])
n_layers = 10
hidden_features = 16

dnumber = 0
device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
print(device)

CNFModel = CNF(n_features,
               context_features = context_features, 
               n_layers = n_layers, 
               hidden_features = hidden_features)

CNFModel.train()
CNFModel = CNFModel.to(device)

optimizer = torch.optim.Adam(CNFModel.parameters(),lr=1e-3)


for i,dS in enumerate(dL):
    x_input,x_cond = dS
    x_input = x_input.to(device)
    x_cond = x_cond.to(device)
    optimizer.zero_grad()

    nll = - CNFModel(x_input, context=x_cond)
    cnf_loss = nll.mean()
    cnf_loss.backward()
    optimizer.step()

    print(cnf_loss.item())
    if (i > 5):
        break

