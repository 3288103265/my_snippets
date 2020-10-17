import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',
                        '--nodes',
                        default=1,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g',
                        '--gpus',
                        default=2,
                        type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr',
                        '--nr',
                        default=0,
                        type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs',
                        default=2,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    # 提前设置参数， os里面的参数 也可以在shell里面设置。export MASTER_ADDR=localhost
    # 不用export也可以啊
    ###########################################################
    args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = '10.57.23.164' #局域网地址
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    mp.spawn(train, nprocs=args.gpus, args=(args, ))
    ###########################################################


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu_id, args):
    ########################在训练时候再次设置###################################
    rank = args.nr * args.gpus + gpu_id
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=rank)
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu_id)
    model.cuda(gpu_id)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    # device_ids
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
    #########################################################################
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=False)

    ######################使用分布式sampler####################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu_id == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, args.epochs, i + 1, total_step, loss.item()))
    # out_list = [torch.zeros_like(images)] * args.world_size
    # tensor = images
    # dist.all_gather(out_list, tensor)
    if gpu_id == 0:
        print("Training complete in: " + str(datetime.now() - start))

    dist.barrier()
    for id in range(args.world_size):
        if gpu_id == id:
            print(
                f'This the {id} process, images with shape {images.shape}, loss: {loss}'
            )

    # inplace all-reduce
    loss_list = [torch.zeros_like(loss) for _ in range(args.world_size)] 
    dist.all_gather(loss_list, loss)
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)

    for id in range(args.world_size):
        if gpu_id == id:
            print(loss_list)
            print(loss.item())


if __name__ == '__main__':
    main()
