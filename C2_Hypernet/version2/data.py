from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, path='C2_Hypernet/version2/mnist_data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1))
    ])
    train_ds = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=1000)
    return train_loader, test_loader
