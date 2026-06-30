from modelClasses import ddp_setup, CNF, CNF_trainer, Encoder
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import sys
import pickle
from pathlib import Path
from glob import glob

from validateCNF import valCNF, generate_seeds
import consts

def prepare_Models(rank : int, hyper_params : dict):  

    e = Encoder(consts.input_dim, consts.middle_dim, consts.output_dim)


    cnf = CNF(n_features = consts.n_features,
              context_features= consts.context_features,
              n_layers = consts.n_layers,
              hidden_features = consts.hidden_features,
              num_bins = consts.num_bins,
              tails = consts.tails,
              tail_bound = consts.tail_bound)

    if rank == 0 :
        temp = {"CNFn_layers" : consts.n_layers,
                "Layer Type" : [],
                "Hidden Width" : consts.hidden_features,
                "spline_num_bins" : consts.num_bins, 
                "spline_tails" : consts.tails, 
                "spline_tail_bound" : consts.tail_bound,
                "E middle_dim" : consts.middle_dim,
                "E output_dim" : consts.output_dim
                }

        uniqueLayers = int(len(cnf.transforms) / consts.n_layers)
       
        temp_idx = 0
        temp_str = str()
     
        for layer_idx in range(uniqueLayers):
            temp_str = str(cnf.transforms[layer_idx])
            temp_idx = temp_str.find("(")
            temp["Layer Type"].append(temp_str[:temp_idx])
        
        hyper_params.update(temp)
        print(f"prepareModel : {hyper_params}")

    return e, cnf

def main(rank: int, world_size: int, total_epochs: int, batch_size: int, base_hyper_params : dict, base_PATH : str):
    hyper_params = dict(base_hyper_params)
    ddp_setup(rank, world_size)
    Emodel, CNFmodel = prepare_Models(rank, hyper_params)
    CNFmodel = CNFmodel.train()
    Emodel = Emodel.train()
    theta_paths = sorted(glob(consts.theta_path))
    data_paths = sorted(glob(consts.data_path))
    val_theta_paths = sorted(glob(consts.val_theta_path))
    val_data_paths = sorted(glob(consts.val_data_path))
    trainer = CNF_trainer(Emodel,CNFmodel, theta_paths, data_paths, val_theta_paths, val_data_paths, rank, batch_size)  

    if rank == 0:
        hyper_params["Optimizer"] = str(trainer.optimizer)
        PATH = base_PATH + "hP.bin" 
        with open(PATH, 'wb') as handle:
            pickle.dump(hyper_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    trainer._train(total_epochs,base_PATH)
    torch.distributed.barrier() #makes sure all GPUs finish before next one starts
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    args = sys.argv
    runVal = args[1]
    CNFName = args[2]

    PATH = f"Models/{CNFName}/"
    Path(PATH).mkdir(parents=True, exist_ok=True)

    CNF_hyperParams = {}

    CNF_hyperParams =  {"batch_size" : consts.batch_size,
             "Grad Clip" : True,
             "Epochs" : consts.total_epochs}

    mp.spawn(main, args=(world_size, consts.total_epochs, consts.batch_size, CNF_hyperParams, PATH), nprocs=world_size) 
    
    print("Finished")
    
    if runVal == "True" :

        thetaMean = np.load(consts.theta_mean_path)
        thetaStd = np.load(consts.theta_std_path) 
        #dataTest = torch.from_numpy(np.load(consts.test_data_path)[:300000])
        #paramsTest = np.load(consts.test_theta_path)[:300000]

        dnumber = 0
        device = torch.device(f"cuda:{dnumber}" if torch.cuda.is_available() else "cpu")
        print(device)

        EModel = Encoder(input_dim = consts.input_dim,
                         middle_dim = consts.middle_dim,
                         output_dim = consts.output_dim)

        CNFModel = CNF(n_features = consts.n_features,
                       context_features = consts.context_features, 
                       n_layers = consts.n_layers, hidden_features = consts.hidden_features, 
                       num_bins = consts.num_bins, tails = consts.tails, 
                       tail_bound = consts.tail_bound) 

        ckpt = torch.load(PATH + "Model_checkpoint.pt", map_location=device)
        CNFModel.load_state_dict(ckpt["CNF_Model"])
        CNFModel.eval()
        CNFModel = CNFModel.to(device)

        EModel.load_state_dict(ckpt["E_Model"])
        EModel.eval()
        EModel = EModel.to(device)

        #valCNF(PATH, EModel, CNFModel, device,
        #        thetaMean,thetaStd,dataTest,paramsTest)
        
        generate_seeds(consts.base_path, 50000, EModel , CNFModel, device, thetaMean, thetaStd)



