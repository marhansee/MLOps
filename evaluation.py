import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import yaml
from model import CustomResNet
import time

def load_model(weight_path, device):
    model = CustomResNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_inference_time = 0.0
    total_samples = 0

    print("Initializing validation")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

    test_loss /= total_samples
    accuracy = 100. * correct / total_samples
    avg_inference_time = (total_inference_time / total_samples) * 1000

    print(f'\nValidation set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)')
    print(f'Average inference time per image: {avg_inference_time:.6f} ms\n')

    os.makedirs('results', exist_ok=True)
    with open('results/results.txt', 'a') as f:
        f.write(f'Accuracy: {accuracy:.2f}%, Avg Inference Time: {avg_inference_time:.6f}s\n')

def main():
    test_config_path = 'test_config.yaml'
    config = load_config(test_config_path)

    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(config['seed'])

    test_kwargs = {'batch_size': config['test_batch_size']}
    if use_cuda:
        test_kwargs.update({'num_workers': 2, 'pin_memory': True, 'shuffle': True})

    test_transform = transforms.Compose([
        transforms.Resize((275, 275)),
        transforms.ToTensor()
    ])

    test_data = datasets.ImageFolder(os.path.join(config['data_dir'], 'test'),
                                     transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    model = load_model(config['weight_path'], device=device)

    validate(model=model, device=device, test_loader=test_loader)

if __name__ == '__main__':
    main()
