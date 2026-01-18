import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("nccl:", torch.cuda.nccl.version() if torch.cuda.is_available() else None)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

