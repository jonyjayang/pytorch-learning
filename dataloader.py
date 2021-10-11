import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10('./dataset',download=True,train=True,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
img,tartget =test_data[0]

write = SummaryWriter('log')
step =0
for data in test_loader:
    imgs,tartgets = data
    write.add_images('test_data',imgs,step)
    step+=1
write.close()