from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = features = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        return x, features 

class Discriminator(nn.Module):
    #first:     input - 200 - 100 - 2
    #deeperD:   input - 400 - 400 - 100 - 2, good in most cases
    #deeperD2:  input - 400 - 400 - 400 - 400 - 2
    def __init__(self):
        super(Discriminator, self).__init__()
        self.operation = nn.Sequential( nn.Linear(20 + 320 + 1, 400), #10+1
                                        nn.ReLU(),
                                        nn.Dropout(),

                                        nn.Linear(400, 400),
                                        nn.ReLU(),
                                        nn.Dropout(),

                                        nn.Linear(400, 100),
                                        nn.ReLU(),
                                        nn.Dropout(),

                                        nn.Linear(100, 2)   )

    def forward(self, x):
        x = self.operation(x)
        return F.log_softmax(x, dim=1)

def reverse_grad_hook(self, grad_input, grad_output):
    assert(len(grad_input) == 3)
    #assert(grad_input[1].size()[1] == 21)#10+1

    return grad_input[0]*1, grad_input[1]*-1, grad_input[2]*1

def train_teacher(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            #test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acy = 1.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acy))
    return acy

def train_student_batch(args, teacher_model, student_model, discriminator, device, train_loader, optimizer, epoch):
    teacher_model.train()
    student_model.train()
    discriminator.train()
    for batch_idx, (data, c_target) in enumerate(train_loader):
        data, c_target = data.to(device), c_target.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_output, teacher_features = teacher_model(data)
        student_output, _ = student_model(data)

        classifier_output = torch.cat((teacher_output, student_output), dim=0) #tensor
        classifier_output = (classifier_output,
                            torch.cat((teacher_features, teacher_features), dim=0)) #tuple
        classifier_output = torch.cat(classifier_output, dim=1) #tensor
        c_target = c_target.float().unsqueeze_(dim=1)*1.0 # int64 to float32, (size) to (size, 1)
        c_target = torch.cat((c_target, c_target), dim=0) #duplicated tensor
        classifier_output = torch.cat((classifier_output, c_target), dim=1)

        d_target = (torch.zeros(data.size()[0], dtype=torch.int64), torch.ones(data.size()[0], dtype=torch.int64))
        d_target = torch.cat(d_target, dim=0).to(device)

        d_output = discriminator(classifier_output)
        loss = F.nll_loss(d_output, d_target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='\t')

            print('Accuracy of D: {:.2f}%'.format( 
                    100.0* (d_output.max(dim=1)[1] == d_target).sum().item() / d_target.size()[0] ))

def train_student_parallel(args, teacher_model, student_model, discriminator, device, train_loader, optimizer, epoch):
    teacher_model.train()
    student_model.train()
    discriminator.train()
    for batch_idx, (data, c_target) in enumerate(train_loader):
        data, c_target = data.to(device), c_target.to(device)
        rand_permute = torch.randperm(data.size()[0]).to(device)
        data, c_target = data[rand_permute], c_target[rand_permute]
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_output, teacher_features = teacher_model(data) 
        student_output, _ = student_model(data)
        assert(data.size()[0]%2==0)
        half_idx = int(data.size()[0]/2)

        classifier_output = torch.cat((teacher_output[:half_idx], student_output[:half_idx], teacher_features[:half_idx]), dim=1) #tensor
        classifier_output = (classifier_output, 
                            torch.cat((student_output[half_idx:], teacher_output[half_idx:], teacher_features[half_idx:]), dim=1) ) #tuple
        classifier_output = torch.cat(classifier_output, dim=0) #tensor
        c_target = c_target.float().unsqueeze_(dim=1)*1.0 #int64 to float32, (size) to (size, 1)
        classifier_output = torch.cat((classifier_output, c_target), dim=1)

        d_target = (torch.zeros(half_idx, dtype=torch.int64), torch.ones(half_idx, dtype=torch.int64))
        d_target = torch.cat(d_target, dim=0).to(device)

        d_output = discriminator(classifier_output)
        loss = F.nll_loss(d_output, d_target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='\t')

            print('Accuracy of D: {:.2f}%'.format( 
                    100.0* (d_output.max(dim=1)[1] == d_target).sum().item() / d_target.size()[0] ))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    teacher_model = Net().to(device)
    optimizer = optim.SGD(teacher_model.parameters(), lr=0.01, momentum=args.momentum)
    print(teacher_model)
    print(optimizer)

    for epoch in range(1, 11):
        train_teacher(args, teacher_model, device, train_loader, optimizer, epoch)
        test(args, teacher_model, device, test_loader)

    ##################################################################
    student_model = Net().to(device)
    discriminator = Discriminator().to(device)
    discriminator.operation[0].register_backward_hook(reverse_grad_hook)
    optimizer = optim.SGD(list(student_model.parameters()) + list(discriminator.parameters()), lr=args.lr, momentum=args.momentum)
    print(student_model)
    print(discriminator)
    print(optimizer)

    best_student_acy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_student_parallel(args, teacher_model, student_model, discriminator, device, train_loader, optimizer, epoch)
        test(args, teacher_model, device, test_loader)
        acy = test(args, student_model, device, test_loader)
        
        if(best_student_acy < acy):
            best_student_acy = acy
        print('Best Student Test Accuracy: {:.2f}%\n'.format(100.0 * best_student_acy))





if __name__ == '__main__':
    main()