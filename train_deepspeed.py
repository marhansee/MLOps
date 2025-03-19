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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
import time

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(model_engine, train_loader):
    model_engine.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model_engine.local_rank).half(), target.to(model_engine.local_rank)

        model_engine.zero_grad()
        
        output = model_engine(data)
        loss = F.cross_entropy(output, target)

        model_engine.backward(loss)
        model_engine.step()  # Replaces optimizer.step()

        total_loss += loss.item()

   
    return total_loss / len(train_loader)


def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            data = data.half() # Ensure data is in FP16

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
    config_path = 'train_ddp_config.yaml'
    config = load_config(config_path)
    torch.manual_seed(config['seed'])


    deepspeed.init_distributed()
    # Initialize WandB
    wandb.login()
    wandb.init(project=config['wandb']['project'], config=config)

    # Initialize ranks and process groups
    rank = dist.get_rank()

    # Define device ID and load model with device
    device_id = rank % torch.cuda.device_count()
    model = CustomResNet().to(device_id)


    train_kwargs = {'batch_size': config['batch_size']}
    test_kwargs = {'batch_size': config['test_batch_size'], 'shuffle': False}

    cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_transform = transforms.Compose([
        transforms.Resize((275, 275)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((275, 275)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    # Datasets and loaders
    train_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'train'),
                                     transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'test'),
                                     transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)


    # Wrap model in DeepSpeed
    ddp_model,_, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config='ds_config.json',
    )


    # Initialize best_loss to a large value
    best_loss = float('inf')


    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        avg_loss = train(ddp_model, train_loader)
        val_loss, val_accuracy = validate(ddp_model, device_id, test_loader)  # Validation step
        
        # Save model if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            if rank == 0:
                torch.save(ddp_model.state_dict(), "best_model_deepspeed.pth")
                print(f"Epoch {epoch}: New best model saved with validation loss {best_loss:.6f}")

        print(f"Epoch {epoch}: Average Train Loss: {avg_loss:.6f}, Validation Loss: {val_loss:.6f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")

if __name__ == '__main__':

        start_time = time.time()
        main()
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Execution time in seconds: {execution_time}")
