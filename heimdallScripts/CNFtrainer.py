from modelClasses import ddp_setup, CNF, CNF_trainer, trainingDataSet , autoEncoder
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import torchinfo
import pickle
from pathlib import Path

from validateCNF import valCNF


EPSILON = 1e-3

data = torch.tensor(np.load("/raid/vigneshk/data/dataTrain.npy")).float()
thetaStandard = torch.tensor(np.load("/raid/vigneshk/data/paramsTrain.npy")).float()

def prepare_Models(rank : int, hyper_params : dict):
    middleRatio = 0.90
    compressRatio = 0.80
    
    input_dim = int(data.shape[1])
    middle_dim = int(data.shape[1]*middleRatio)
    output_dim = int(data.shape[1]*compressRatio)

    n_features = int(thetaStandard.shape[1])
    n_layers = 8
    hidden_features = 25
    contextF = output_dim
    num_bins = 16
    tails = "linear"
    tail_bound = 3.5

    ae = autoEncoder(input_dim, middle_dim, output_dim)


    cnf = CNF(n_features,
              context_features= contextF,
              n_layers=n_layers,
              hidden_features=hidden_features,
              num_bins = num_bins,
              tails = tails,
              tail_bound = tail_bound)

    if rank == 0 :
        temp = {"CNFn_layers" : n_layers,
                "Layer Type" : [],
                "Hidden Width" : hidden_features,
                "spline_num_bins" : num_bins, 
                "spline_tails" : tails, 
                "spline_tail_bound" : tail_bound,
                "AE middle_dim" : middle_dim,
                "AE output_dim" : output_dim
                }

        uniqueLayers = int(len(cnf.transforms) / n_layers)
       
        temp_idx = 0
        temp_str = str()
     
        for layer_idx in range(uniqueLayers):
            temp_str = str(cnf.transforms[layer_idx])
            temp_idx = temp_str.find("(")
            temp["Layer Type"].append(temp_str[:temp_idx])
        
        hyper_params.update(temp)
        print(f"prepareModel : {hyper_params}")

    return ae, cnf

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset, shuffle = True)
    )

def main(rank: int, world_size: int, total_epochs: int, batch_size: int, base_hyper_params : dict, base_PATH : str):
    hyper_params = dict(base_hyper_params)
    ddp_setup(rank, world_size)
    AEmodel, CNFmodel = prepare_Models(rank, hyper_params)
    CNFmodel = CNFmodel.train()
    AEmodel = AEmodel.train()
    dataset = trainingDataSet(thetaStandard,data)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = CNF_trainer(AEmodel,CNFmodel, train_data, rank, batch_size)  

    if rank == 0:
        hyper_params["Optimizer"] = str(trainer.optimizer)
        PATH = base_PATH + "hP.bin" 
        #print(f"main : {hyper_params}")


        with open(PATH, 'wb') as handle:
            pickle.dump(hyper_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    trainer._train(total_epochs,base_PATH)
    torch.distributed.barrier() #makes sure all GPUs finish before next one starts
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 512
    total_epochs = 30

    args = sys.argv
    runVal = args[1]
    CNFName = args[2]

    PATH = f"/raid/vigneshk/Models/{CNFName}/"
    Path(PATH).mkdir(parents=True, exist_ok=True)

    CNF_hyperParams = {}

    CNF_hyperParams =  {"batch_size" : batch_size,
             "Grad Clip" : True,
             "Epochs" : total_epochs}

    print(f"Before : {CNF_hyperParams}")
    
    mp.spawn(main, args=(world_size, total_epochs, batch_size, CNF_hyperParams, PATH), nprocs=world_size) 
    
    print("Finished")
    
    if runVal == "True" :
        print(runVal)
        valCNF(PATH,1)



