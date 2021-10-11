from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    write = SummaryWriter('log')
    for i in range(10):
        write.add_scalar('y=3x',3*i,i)
    write.close()
