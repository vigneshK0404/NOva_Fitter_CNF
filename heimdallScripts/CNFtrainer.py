from modelClasses import ddp_setup, CNF, CNF_trainer, trainingDataSet
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


thetaData = torch.tensor(np.load("/raid/vigneshk/data/theta_data.npy")).float()
dataPoisson_latent = torch.tensor(np.load("/raid/vigneshk/data/poissonEncoded.npy")).float()
#thetaStandard = np.load(dataBase + "thetaData_standard.npy")

def prepare_Model():
    n_features = int(thetaData.shape[1])
    n_layers = 10
    hidden_features = 16
    contextF = int(dataPoisson_latent.shape[1])

    print(f"data:{n_features}, context:{contextF}")
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
    dataset = trainingDataSet(thetaData,dataPoisson_latent)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = CNF_trainer(CNFmodel, train_data, rank, batch_size)
    trainer._train(total_epochs)
    trainer._save_checkpoint()
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 1024
    total_epochs = 5
    mp.spawn(main, args=(world_size, total_epochs, batch_size), nprocs=world_size)





