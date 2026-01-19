from modelClasses import *


base = "/raid/vigneshk/"
dataBase = base + "data/"
dataPoisson = np.load(dataBase + "poissonData.npy")
thetaData = np.load(dataBase + "theta_data.npy")
thetaStandard = np.load(dataBase + "thetaData_standard.npy")

def prepare_Model():
    input_dim = int(dataPoisson.shape[1])
    middle_dim = 32
    output_dim = int(input_dim / 2)

    #print(f"input:{input_dim}, output_dim:{output_dim}")
    return autoEncoder(input_dim, middle_dim, output_dim)



def prepare_dataloader(dataset: torch.tensor, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    AEmodel = prepare_Model()
    AEmodel = AEmodel.train()
    dataset = torch.tensor(dataPoisson).float()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = AE_trainer(AEmodel, train_data, rank, batch_size)
    trainer._train(total_epochs)
    #trainer._save_checkpoint()
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 4096
    total_epochs = 10
    mp.spawn(main, args=(world_size, total_epochs, batch_size), nprocs=world_size)


#print(AEmodel.r_losses[-1])
'''
testP,_ ,_= generateTrainingData(1,1)
testP = torch.tensor(testP, dtype = torch.float32)
testP = testP.to(device)
decodePtest = encodeModel(testP).to("cpu").detach().numpy().flatten()
testP_CPU = testP.to("cpu").detach().numpy().flatten()

print(decodePtest)
print(testP_CPU)

bins = np.arange(0,20.5,0.5)
plt.hist(bins[:-1], bins,weights=testP_CPU,color='blue',edgecolor='black',alpha=0.5,label="TrueBin")
plt.hist(bins[:-1], bins,weights=decodePtest,color='red',edgecolor='black',alpha=0.4,label="decodedBin")
plt.legend()
'''
