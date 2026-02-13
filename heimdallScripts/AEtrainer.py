from modelClasses import ddp_setup, autoEncoder, AE_trainer
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys



dataPoisson_scaled = np.load("/raid/vigneshk/data/dP_scaled.npy") 
#np.load("/raid/vigneshk/data/poissonData.npy")

def prepare_Model():
    input_dim = int(dataPoisson_scaled.shape[1])
    middle_dim = int(input_dim / 2)
    output_dim = int(input_dim / 3)

    print(f"input:{input_dim} \n middle_dim:{middle_dim} \n output_dim:{output_dim}")
    return autoEncoder(input_dim, middle_dim, output_dim)



def prepare_dataloader(dataset: torch.tensor, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset,shuffle=True)
    )

def main(rank: int, world_size: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    AEmodel = prepare_Model()
    AEmodel = AEmodel.train()
    dataset = torch.tensor(dataPoisson_scaled).float()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = AE_trainer(AEmodel, train_data, rank, batch_size)
    trainer._train(total_epochs) 
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 1024
    total_epochs = 15

    #TODO Add argv here for latent Gen

    mp.spawn(main, args=(world_size, total_epochs, batch_size), nprocs=world_size)

    if runGen == "True":


