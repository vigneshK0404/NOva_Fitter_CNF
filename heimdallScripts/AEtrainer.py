from modelClasses import ddp_setup, autoEncoder, AE_trainer
from generateLatent import latentGen
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
from pathlib import Path


def prepare_Model(data_scaled : np.array):
    input_dim = int(data_scaled.shape[1])
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

def main(rank: int, world_size: int, total_epochs: int, batch_size: int, PATH : str, data_scaled : np.array):
    ddp_setup(rank, world_size)
    AEmodel = prepare_Model(data_scaled)
    AEmodel = AEmodel.train()
    dataset = torch.tensor(data_scaled).float()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = AE_trainer(AEmodel, train_data, rank, batch_size)
    trainer._train(total_epochs, PATH) 
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 1024
    total_epochs = 15

    folderName = sys.argv[2]

    PATH = f"/raid/vigneshk/Models/{folderName}/"
    Path(PATH).mkdir(parents=True, exist_ok=True)
    data_scaled = np.load("/raid/vigneshk/data/dataTrain.npy")


    mp.spawn(main, args=(world_size, total_epochs, batch_size, PATH, data_scaled), nprocs=world_size)

    if sys.argv[1] == "True":
        latentGen(PATH, data_scaled)



