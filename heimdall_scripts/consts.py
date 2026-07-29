from pathlib import Path
import getpass
#Change hyper-params here

#Standardizing_Params
EPSILON = 1e-3
uniqueSample = 100000
repeatSize = 150
dnumber = 0

#Encoder_Params
input_dim = 148
middleRatio = 1.5
compressRatio = 2
middle_dim = int(input_dim * middleRatio)
output_dim = int(input_dim * compressRatio)

#CNF_Params
n_features = 6
n_layers = 12
hidden_features = 30
context_features = output_dim
num_bins = 24
tails = "linear"
tail_bound = 3.5


#Training_Params
batch_size = 4096
total_epochs = 12
encoder_lr = 1e-3
cnf_lr = 1e-3

#Standardizing Paths
USERNAME = getpass.getuser()

CODE_ROOT = Path("/raid") / USERNAME
DATA_ROOT = Path("/share/lazy") / USERNAME / "data"

MODELS = CODE_ROOT / "Models"
INFERENCE_DATA = CODE_ROOT / "inference_data"

RAW_DATA = DATA_ROOT / "raw_data"
RAW_DATA_TRAINING = RAW_DATA / "training"
RAW_DATA_VAL = RAW_DATA / "validation"
PROCESSED_DATA = DATA_ROOT / "processed"
PROCESSED_DATA_TRAINING = PROCESSED_DATA / "training"
PROCESSED_DATA_VAL = PROCESSED_DATA / "validation"
PROCESSED_DATA_STATS = PROCESSED_DATA / "stats"

TMP_ROOT_FILES = DATA_ROOT / "tmp_root_files"

theta_mean_path = str(PROCESSED_DATA_STATS / "theta_mean.npy")
data_mean_path = str(PROCESSED_DATA_STATS / "data_mean.npy")
theta_std_path = str(PROCESSED_DATA_STATS / "theta_std.npy")
data_std_path = str(PROCESSED_DATA_STATS / "data_std.npy")

data_path = str(PROCESSED_DATA_TRAINING / "*_data_*")
theta_path = str(PROCESSED_DATA_TRAINING / "*_theta_*")
val_data_path = str(PROCESSED_DATA_VAL / "*_data_*")
val_theta_path = str(PROCESSED_DATA_VAL / "*_theta_*")


def create_workspace():

    dirs = [MODELS, INFERENCE_DATA, RAW_DATA_TRAINING, 
            RAW_DATA_VAL, PROCESSED_DATA_TRAINING, 
            PROCESSED_DATA_VAL, PROCESSED_DATA_STATS , TMP_ROOT_FILES]

    for d in dirs :
        d.mkdir(parents=True, exist_ok=True)

    print(f"Workspace created at {CODE_ROOT}")
    print(f"Data-space created at {DATA_ROOT}")


if __name__ == "__main__":
    create_workspace()
    

