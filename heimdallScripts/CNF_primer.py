from modelClasses import CNF, trainingDataSet

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
thetaData = torch.tensor(np.load("/raid/vigneshk/data/theta_data.npy")).float()


dataMap = trainingDataSet(thetaData,dataPoisson_latent)

dL = DataLoader(dataMap, batch_size = 1024, shuffle = True)


n_features = int(thetaData.shape[1])
context_features = int(dataPoisson_latent.shape[1])
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

optimizer = torch.optim.Adam(CNFModel.parameters(),lr=5e-5)


for x_input,x_cond in dL:
    x_input = x_input.to(device)
    x_cond = x_cond.to(device)
    optimizer.zero_grad()

    nll = - CNFModel(x_input, context=x_cond)
    cnf_loss = nll.mean()
    cnf_loss.backward()
    optimizer.step()

    print(cnf_loss.item())
    break


