import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb
import os
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.training.supervised import Naive
from avalanche.logging import TextLogger
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin, EWCPlugin
from avalanche.benchmarks.classic import SplitMNIST
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


def main():

    # Start new wandb run
    wandb.login()
    wandb.init(project="MLOps_project_k8")

    # Create folder for snapshot
    os.makedirs(f"snapshots", exist_ok=True)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save_model_name', type=str, default="mnist_cnn_0_4.pt")
    parser.add_argument('--train_loader_name', type=str, default="train_loader_5_9")
    args = parser.parse_args()

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set random state
    torch.manual_seed(args.seed)

    # Split MNIST
    benchmark = SplitMNIST(n_experiences=2, return_task_id=False)  # First task (0-4), second task (5-9)

    # Initialize model
    model = Net().to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Initialize memory buffer
    replay_plugin = ReplayPlugin(mem_size=1000)  # Stores 1000 samples from past tasks
    ewc = EWCPlugin(ewc_lambda=0.001)

    # Set up wandb logger
    logger = TextLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True),
        loss_metrics(epoch=True, experience=True),
        forgetting_metrics(experience=True),
        loggers=[logger]
    )

    # Define Avalanche training strategy
    strategy = Naive(
        model, optimizer,
        train_mb_size=args.batch_size, eval_mb_size=args.test_batch_size, device=device,
        plugins=[replay_plugin, ewc],
        evaluator=eval_plugin
    )


    # Train and evaluate model
    for experience in benchmark.train_stream:
        print(f"Training on Experience {experience.current_experience}")

        for epoch in range(args.epochs+1):
            print(f"Epoch {epoch}/{args.epochs}")
            
            strategy.train(experience)
            strategy.eval(benchmark.test_stream)

            scheduler.step()

            # Log accuracy for each task separately
            task_accuracy = eval_plugin.get_all_metrics()

            wandb.log({"0_4_accuracy": task_accuracy["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"][1][0]})
            wandb.log({"0_4_loss": task_accuracy["Loss_Exp/eval_phase/test_stream/Task000/Exp000"][1][0]})
            wandb.log({"5_9_accuracy": task_accuracy["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001"][1][0]})
            wandb.log({"5_9_loss": task_accuracy["Loss_Exp/eval_phase/test_stream/Task000/Exp001"][1][0]})

    # Save model
    if args.save_model:
        torch.save(model.state_dict(), f"snapshots/{args.save_model_name}")

    wandb.finish()


if __name__ == '__main__':
    main()
