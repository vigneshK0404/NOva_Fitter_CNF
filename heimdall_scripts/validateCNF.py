from modelClasses import CNF, Encoder
from readCNFROOT import applyStd
import consts

import torch
import numpy as np
from tqdm import tqdm
import uproot
from sklearn.cluster import MeanShift
import pickle
from fpdf import FPDF


def ModeMeanShift(thetaDist: np.array, smoothing: float, minRatio: int):

    min_freq = max(5, len(thetaDist) // 10000)
    bandwidth = estimate_bandwidth(thetaDist, quantile=0.05, n_samples=min(len(thetaDist), 2000)) * smoothing
    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True,
        max_iter=300,
        min_bin_freq=min_freq,
        cluster_all = False
    )

    labels = ms.fit_predict(thetaDist)


    clusters = []

    for i in np.unique(labels):
        mask = (labels == i)
        clusters.append(thetaDist[mask])

    return clusters

def write_architecture(base_PATH : str):
    PATH = base_PATH + "hP.bin"
    with open(PATH, 'rb') as handle:
        hyper_params = pickle.load(handle)

    full_string = str()
   
    for key,value in hyper_params.items():
        full_string += key
        full_string += " : "
        full_string += str(value)
        full_string += "\n"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.multi_cell(w=0,h=10,txt=full_string,border=1)
    pdf.output(base_PATH + "Architecture.pdf","F")


def generate_seeds(model_PATH : str, NumSamples : int, 
                  EModel : Encoder, CNFModel : CNF, device, 
                  thetaMean, thetaStd):

    
    write_architecture(model_PATH)

    file = uproot.open(consts.INFERENCE_DATA / "diagnoseData.root")
    tree = file["dataTree"]
    branches = tree.arrays()
    data = np.array(branches["data"],dtype=np.int32)
    data = torch.tensor(applyStd(data), device=device).float()

    rep_list = []
    total_len = 0
    len_list = []

    with torch.no_grad():
        x_en = EModel(data)
        samples = CNFModel.flow.sample(NumSamples,context=x_en).cpu()
        sample_cut = samples.reshape(-1,samples.shape[-1]).numpy()
        
        #print(x_en.shape)
        #print(true_best_value.shape)
        best_log = CNFModel(true_best_value,x_en)
        
        data_bunches = np.array_split(sample_cut,len(data))
        theta_bunches = torch.split(x_en,1) 

        for true_theta, bunch in tqdm(zip(theta_bunches, data_bunches)):
            assert len(bunch) == NumSamples, len(bunch)
            assert len(true_theta) == 1, len(true_theta)

            representatives = []
            clusters = ModeMeanShift(bunch, 0.6, 1000)
            cluster_len = len(clusters)
            len_list.append(cluster_len)
            total_len += cluster_len

            print(f"Num Clusters : {cluster_len}")

            for cluster in clusters:
                #print(cluster.shape)
                kSamples = cluster.shape[0]
                cluster = torch.tensor(cluster,device=device).float()
                x_en_Exp = true_theta.unsqueeze(1).expand(1,kSamples,-1).reshape(kSamples,-1)
                firstPass = CNFModel(cluster,x_en_Exp)
                mask = np.isfinite(firstPass)
                firstPass = firstPass[mask]
                infer = cluster[torch.argmax(firstPass)]
                representatives.append(infer)   
   
            reps = torch.stack(representatives)
            x_en_reps = true_theta.unsqueeze(1).expand(1,len(reps),-1).reshape(len(reps),-1)
            log_vals = CNFModel(reps,x_en_reps)
            logRankings = log_vals.argsort(descending=True)
            print(f"logRankings : {log_vals[logRankings]}")
            print(f"bestLog : {best_log}")
            rep_list.append(np.asarray(reps[logRankings].cpu()))

            
    final_reps = np.concatenate(rep_list)
    final_reps *= (thetaStd + consts.EPSILON)
    final_reps += thetaMean
    final_reps = final_reps.astype(np.float32).reshape(total_len,-1)
    print(final_reps.shape)

    len_arr = np.array(len_list)
    ncols = len(data)

    if len_arr.ndim == 1:
        len_arr = len_arr.reshape(1, -1)

    assert len_arr.shape[1] == ncols, len_arr.shape
    print(len_arr)

    with uproot.recreate(consts.INFERENCE_DATA / "cnfpreds_diagnose.root") as f:
        f["tree"] = {"reps": final_reps}

        f.mktree("lens", {"lens": np.dtype((np.int16, (ncols,)))})
        f["lens"].extend({"lens": len_arr})
   
    return                


if __name__ == "__main__":

    model_PATH = "Models/NOvACNF_RedOnPlat_lr/"
    thetaMean = np.load(consts.theta_mean_path)
    thetaStd = np.load(consts.theta_std_path) 

    device = torch.device(f"cuda:{consts.dnumber}" if torch.cuda.is_available() else "cpu")
    print(device) 

    EModel = Encoder(input_dim = consts.input_dim,
                          middle_dim = consts.middle_dim,
                          output_dim = consts.output_dim)

    CNFModel = CNF(n_features = consts.n_features,
                   context_features = consts.context_features, 
                   n_layers = consts.n_layers, hidden_features = consts.hidden_features, 
                   num_bins = consts.num_bins, tails = consts.tails, 
                   tail_bound = consts.tail_bound) 

    ckpt = torch.load(model_PATH + "Model_checkpoint.pt", map_location=device)
    CNFModel.load_state_dict(ckpt["CNF_Model"])
    CNFModel.eval()
    CNFModel = CNFModel.to(device)

    EModel.load_state_dict(ckpt["E_Model"])
    EModel.eval()
    EModel = EModel.to(device)
        
    generate_seeds(model_PATH, 50000, EModel , CNFModel, device, thetaMean, thetaStd)
    

