import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            wandb.log({f"Loss": loss.item()})
            if args.dry_run:
                break

def forget_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        # Create masks for forget and retain samples
        forget_mask = (target == 7)
        retain_mask = ~forget_mask
        
        # Initialize losses
        total_loss = torch.tensor(0.0, device=device)
        
        # Forgetting mechanism (for class 7)
        if forget_mask.any():
            forget_output = output[forget_mask]
            original_loss = F.nll_loss(forget_output, target[forget_mask])
            
            # Controlled ascent with clipping
            loss_forget = -torch.clamp(original_loss, max=1.0) 

            total_loss += loss_forget
        
        # Retention mechanism (for other classes)
        if retain_mask.any():
            loss_retain = F.nll_loss(output[retain_mask], target[retain_mask])
            total_loss += loss_retain
        
        total_loss.backward()
        optimizer.step()
        
        # Logging
        if batch_idx % args.log_interval == 0:
            log_data = {
                'epoch': epoch,
                'total_loss': total_loss.item(),
            }
            
            # Add component losses if they exist
            if forget_mask.any():
                log_data['forget_loss'] = loss_forget.item()
            if retain_mask.any():
                log_data['retain_loss'] = loss_retain.item()
            

            wandb.log(log_data)
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = [0] * 10  # 10 classes
    class_total = [0] * 10  # 10 classes

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            # Get the predicted class
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Update per-class correct counts and total counts
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += pred[i].eq(target[i]).item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Calculate and print per-class accuracy
    for i in range(10):  # 10 classes
        if class_total[i] > 0:
            per_class_accuracy = 100. * class_correct[i] / class_total[i]
            print(f'Accuracy for class {i}: {per_class_accuracy:.2f}%')
            wandb.log({f"Accuracy for class {i}": per_class_accuracy})

def main():
    wandb.login()
    wandb.init(project="MLOps_Unlearning")


    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--weights_path', type=str, default="snapshots/baseline.pt")
    parser.add_argument('--save_model_name', type=str, default="baseline.pt")


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    model = Net().to(device)
    
    # Load model weights
    if args.load_model == True:
        model.load_state_dict(torch.load(args.weights_path))


    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


    for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, epoch) # Normal training
        forget_train(args, model, device, train_loader, optimizer, epoch) # Unlearning
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), f"snapshots/{args.save_model_name}")


if __name__ == '__main__':
    main()