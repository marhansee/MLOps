import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
import wandb
import os


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
            if args.dry_run:
                break

        print(f"Train Epoch: {epoch}, Loss: {loss.item()}")
        if batch_idx % 500 == 0:
            wandb.log({f"Loss": loss.item()})




def test(model, device, test_loader, loader_name):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            batch_loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += batch_loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct

            batch_size = len(data)
            total_samples += batch_size

            if (batch_idx + 1) % 500 == 0:
                batch_accuracy = 100. * correct / total_samples
                print(f'Batch {batch_idx + 1}: Accuracy: {batch_accuracy:.2f}%')
                wandb.log({f"{loader_name} Accuracy (Batch {batch_idx+1})": batch_accuracy})

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    wandb.log({f"{loader_name} Loss": test_loss, f"{loader_name} Accuracy": accuracy})


def filter_dataset(dataset, labels):
    indices = [i for i, (img, label) in enumerate(dataset) if label in labels]
    return Subset(dataset, indices)

def main():

    # Start new wandb run
    wandb.login()
    wandb.init(project="MLOps_project_k8")

    # Create folder for snapshot
    os.makedirs(f"snapshots", exist_ok=True)

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
    parser.add_argument('--weights_path', type=str, default="snapshots/mnist_cnn.pt")
    parser.add_argument('--save_model_name', type=str, default="mnist_cnn_0_4.pt")
    parser.add_argument('--train_loader_name', type=str, default="train_loader_5_9")

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
    
    # Load MNIST dataset
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, 
                        transform=transform)
    full_test_dataset = datasets.MNIST('../data', train=False, 
                        transform=transform)

    train_dataset_0_4 = filter_dataset(full_train_dataset, [0, 1, 2, 3, 4])
    train_dataset_5_9 = filter_dataset(full_train_dataset, [5, 6, 7, 8, 9])

    test_dataset_0_4 = filter_dataset(full_test_dataset, [0, 1, 2, 3, 4])
    test_dataset_5_9 = filter_dataset(full_test_dataset, [5, 6, 7, 8, 9])


    # Define loaders
    train_loader_0_4 = DataLoader(train_dataset_0_4, **train_kwargs)
    train_loader_5_9 = DataLoader(train_dataset_5_9, **train_kwargs)
    test_loader_0_4 = DataLoader(test_dataset_0_4, **test_kwargs)
    test_loader_5_9 = DataLoader(test_dataset_5_9, **test_kwargs)

    model = Net().to(device)

    # Load model weights
    if args.load_model == True:
        model.load_state_dict(torch.load(args.weights_path))


    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    if args.train_loader_name == 'train_loader_5_9':
        train_loader = train_loader_5_9
    else:
        train_loader = train_loader_0_4

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch) # Train on 5-9 digits
        test(model, device, test_loader_0_4, loader_name="0_to_4") # Evaluate performance for 0-4 
        test(model, device, test_loader_5_9, loader_name="5_to_9") # EValuate performance for 5-9

        scheduler.step()
    
    if args.save_model:
        torch.save(model.state_dict(), f"snapshots/{args.save_model_name}")


if __name__ == '__main__':
    main()
