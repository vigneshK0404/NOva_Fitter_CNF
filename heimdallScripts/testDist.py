import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank : int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


#A = torch.rand(10,6)
base = "/raid/vigneshk/"
dataBase = base + "data/"
dataPoisson = np.load(dataBase + "poissonData.npy")
#print(A,B,sep="\n")


class autoEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, gpu_id : int):
        super().__init__()
        self.gpu_id = gpu_id

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim,middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim,output_dim)
            )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(output_dim,middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim,input_dim),
            )

    def encode(self,x):
        return self.encoder(x)

    def decode(self,x):
        return self.decoder(x)

    def forward(self,x):
        eData = self.encode(x)
        dData = self.decode(eData)
        return dData


def prepare_dataloader(dataset: torch.tensor, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

#max_epochs 5
#batch_size 4096

def main(rank: int, world_size: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset = torch.tensor(dataPoisson).float()
    model = autoEncoder(20,10,16,rank).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    train_data = prepare_dataloader(dataset, batch_size)
    rloss = torch.nn.MSELoss()

    r_losses = []

    for iter in tqdm(range(total_epochs)):
        iter_losses = []
        train_data.sampler.set_epoch(iter)
        for x_batch in train_data:
            x_batch = x_batch.to(rank, non_blocking=True).float()
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = rloss(y_pred, x_batch)
            iter_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        r_losses.append(np.mean(np.array(iter_losses)))

    plt.plot(r_losses)
    plt.savefig("/raid/vigneshk/plots/losses.png")
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    total_epochs = 5
    batch_size = 4096

    mp.spawn(main, args=(world_size, total_epochs, batch_size), nprocs=world_size)
