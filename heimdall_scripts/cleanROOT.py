import uproot
import numpy as np
import glob
from tqdm import tqdm
import os

if __name__ == "__main__":        

    files = sorted(glob.glob("data/*.root"))
    dataList = []
    paramList = []
    failList = []

    idx = 0    
    for file_name in tqdm(files):
               
        with uproot.open(file_name) as f:
            try:
                tree = f["Experimental_Data_Tree"]
            except Exception as e:
                failList.append(file_name)
                os.remove(file_name)
                continue
            
            branches = tree.arrays()

        data = np.array(branches["data"],dtype = np.int32)
        params = np.array(branches["params"], dtype = np.float32)
        dataList.append(data)
        paramList.append(params)

        if len(dataList) == 10:
            data_stack = np.vstack(dataList)
            param_stack = np.vstack(paramList)

            np.savez(f"data/merged/{idx}",data = data_stack, params = param_stack)
            idx += 1
            print(f"Stacked {len(dataList)} root files, saved merged/{idx}.npz")
                            
            dataList = []
            paramList = []

            del data_stack, param_stack

    if len(dataList) > 0:
        data_stack = np.vstack(dataList)
        param_stack = np.vstack(paramList)
        np.savez(f"data/merged/{idx}",data = data_stack, params = param_stack)
        print(f"Stacked {len(dataList)} root files, saved merged/{idx}.npz")

    print(failList)



    
