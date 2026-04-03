import numpy as np
import uproot



def getSterileData(path : str):
    file = uproot.open(path)
    tree = file["Experimental_Data_Tree"]
    branches = tree.arrays()

    print(tree.keys())

    for key in tree.keys():
        print(branches[key])

if __name__ == "__main__":
    getSterileData("data/CNFData.root")

