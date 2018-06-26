import torch
import torch.distributed as dist
import torch.nn.functional as F
from data_utils import partition_dataset

import hps


def train(worker, n_workers):
    torch.manual_seed(hps.seed)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    model.cuda(worker)

    num_batches = int(len(train_set.dataset) / float(bsz))
    for epoch in range(hps.n_epochs):
        epoch_loss = 0.0
        for data, target in train_set:
            data = data.cuda(worker)
            target = target.cuda(worker)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)


def average_gradients(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
