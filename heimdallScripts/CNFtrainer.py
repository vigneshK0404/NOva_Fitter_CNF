from modelClasses import ddp_setup, CNF, CNF_trainer, trainingDataSet
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

EPSILON = 1e-6


#thetaData = torch.tensor(np.load("/raid/vigneshk/data/theta_data.npy")).float()
dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
thetaStandard = torch.tensor(np.load("/raid/vigneshk/data/thetaData_standard.npy")).float()

latent_mean = dataPoisson_latent.mean(axis=0)
latent_std = dataPoisson_latent.std(axis=0, correction=0)
dataPoisson_latent_Standard = (dataPoisson_latent - latent_mean) / (latent_std + EPSILON)





def prepare_Model():
    n_features = int(thetaStandard.shape[1])
    n_layers = 10
    hidden_features = 16
    contextF = int(dataPoisson_latent_Standard.shape[1])

    #print(f"data:{n_features}, context:{contextF}")
    return CNF(n_features,
               context_features= contextF,
               n_layers=n_layers,
               hidden_features=hidden_features)



def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    CNFmodel = prepare_Model()
    CNFmodel = CNFmodel.train()
    dataset = trainingDataSet(thetaStandard,dataPoisson_latent_Standard)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = CNF_trainer(CNFmodel, train_data, rank, batch_size)
    trainer._train(total_epochs)

    torch.distributed.barrier() #makes sure all GPUs finish before next one starts
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 1024
    total_epochs = 7
    mp.spawn(main, args=(world_size, total_epochs, batch_size), nprocs=world_size)





