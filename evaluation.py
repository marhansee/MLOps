import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import yaml
from model import CustomResNet

def load_model(weight_path, device):
    model = CustomResNet().to(device)
    model.load_state_dict(weight_path)

    return model

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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

    with open('results/results.txt', 'a') as f:
        f.write(f'Accuracy score: {accuracy} \n')

def main():
    test_config_path = 'test_config.yaml'
    config = load_config(test_config_path)

    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(config['seed'])

    test_kwargs = {'batch_size': config['test_batch_size']}

    if use_cuda:
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
        test_kwargs.update(cuda_kwargs)

    test_transform = transforms.Compose([
        transforms.Resize((275, 275)),
        transforms.ToTensor()
    ])
    
    test_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'test'),
                                     transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    model = load_model(config['weight_path'])
    
    validate(
        model=model,
        device=device,
        test_loader=test_loader
    )

if __name__ == '__main__':
    main()