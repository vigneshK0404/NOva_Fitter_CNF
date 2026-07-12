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

#Standardizing_Paths
base_path = "/share/lazy/vigneshk/data/"


#Testing_Paths
processed_path = f"{base_path}processed/"
data_path = f"{processed_path}training/*_data_*"
theta_path = f"{processed_path}training/*_theta_*"
val_data_path = f"{processed_path}testing/*_data_*"
val_theta_path = f"{processed_path}testing/*_theta_*"
theta_mean_path = f"{processed_path}stats/theta_mean.npy"
theta_std_path = f"{processed_path}stats/theta_std.npy"
data_mean_path = f"{processed_path}stats/data_mean.npy"
data_std_path = f"{processed_path}stats/data_std.npy"
test_data_path = f"{processed_path}testing/17_data_0.npy"
test_theta_path = f"{processed_path}testing/17_theta_0.npy"

