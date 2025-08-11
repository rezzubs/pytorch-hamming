import torch
import torchvision
from torch import nn
from torchvision import transforms

from pytorch_ecc import HammingStats

model: nn.Module = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
)  # type: ignore


mean = (0.49139968, 0.48215827, 0.44653124)
std = (0.24703233, 0.24348505, 0.26158768)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
dataloader = torch.utils.data.DataLoader(
    testset,
    batch_size=8,
    shuffle=False,
)


def evaluate(model: nn.Module):
    global dataloader
    model.eval()
    num_samples = torch.tensor(0)
    num_correct = torch.tensor(0)

    for data in dataloader:
        inputs, targets = data[0], data[1]

        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()


HammingStats.eval(model, 0.0002, evaluate).summary()
