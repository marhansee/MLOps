import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import wandb
from torch.optim.lr_scheduler import StepLR
from thop import profile, clever_format
from model import CustomResNet
import yaml


def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        return config

def profile_model(model, input_size):
    device = next(model.parameters()).device
    sample_input = torch.randn(*input_size).to(device)
    macs, params = profile(model, inputs=(sample_input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"model profiling: \nMACS:{macs}\nParams: {params}")



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            wandb.log({'Train Loss': loss.item()})


    return epoch_loss / len(train_loader)

def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nValidation set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    # Log validation metrics
    wandb.log({'Validation Loss': test_loss, 'Validation Accuracy': accuracy})

    return test_loss, accuracy

def main():
    config_path = 'train_config.yaml'
    config = load_config(config_path)


    """
    Commented arguments.
    Remove the commented arguments when yaml-config has been tested and works.
    """
    # parser = argparse.ArgumentParser(description='Miniproject DL')

    # parser.add_argument('--batch-size', type=int, default=16, metavar='N',
    #                     help='input batch size for training (default: 16)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=10, metavar='N',
    #                     help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
    #                     help='learning rate (default: 1e-4)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--data-dir', type=str, default='archive',
    #                     help='path to the dataset directory (default: archive)')
    # args = parser.parse_args()

    # Initialize WandB
    wandb.login()
    wandb.init(project=config['wandb']['project'], config=config)

    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(config['seed'])

    train_kwargs = {'batch_size': config['batch_size']}
    test_kwargs = {'batch_size': config['test_batch_size']}

    if use_cuda:
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_transform = transforms.Compose([
        transforms.Resize((275, 275)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    # Datasets and loaders
    train_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'train'), 
                                     transform=test_transform)
    test_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'test'), 
                                     transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    # Model, optimizer, and scheduler
    model = CustomResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer=optimizer,
                       step_size=config['scheduler']['step_size'], 
                       gamma=config['scheduler']['gamma'])
    
    #profile_model(model, input_size=(1, 3, 275, 275))

    # Initialize best_loss to a large value
    best_loss = float('inf')


    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        avg_loss = train(config, model, device, train_loader, optimizer, epoch)
        val_loss, val_accuracy = validate(model, device, test_loader)  # Validation step
        scheduler.step()
        # Save model if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Epoch {epoch}: New best model saved with validation loss {best_loss:.6f}")

        print(f"Epoch {epoch}: Average Train Loss: {avg_loss:.6f}, Validation Loss: {val_loss:.6f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%, Learning Rate: {scheduler.get_last_lr()[0]}")

if __name__ == '__main__':
        main()
