import torch, torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import tifffile
import argparse

# Data augmentation and normalization for training
# Just normalization for validation
class ExtractRGB:
    def __call__(self, tif):
        # EuroSAT bands are 1-indexed, PyTorch tensors are 0-indexed
        # Band 4 (Red), Band 3 (Green), Band 2 (Blue)
        return tif[(3, 2, 1), :, :]

data_transforms = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
def tiff_loader(path):
    image = tifffile.imread(path)
    # EuroSAT bands are 1-indexed, PyTorch tensors are 0-indexed
    # Band 4 (Red), Band 3 (Green), Band 2 (Blue)
    image = image[:, :, (3, 2, 1)]
    image = torch.from_numpy((image/65535*25).astype(np.float32).transpose(2,0,1))
    return torch.nn.functional.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the EuroSAT dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet18().to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    cudnn.benchmark = True

    image_dataset = datasets.ImageFolder(args.dataset, data_transforms, loader=tiff_loader)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes

    train_image_dataset, val_image_dataset, test_image_dataset = torch.utils.data.random_split(image_dataset, [20000, 6900, 100], generator=torch.Generator().manual_seed(42))

    dataloaders = {'train': torch.utils.data.DataLoader(train_image_dataset, batch_size=64, shuffle=True, num_workers=4), 'val': torch.utils.data.DataLoader(val_image_dataset, batch_size=4, shuffle=True, num_workers=4)}
    dataset_sizes = {'train': len(train_image_dataset), 'val': len(val_image_dataset)}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss() # Mean Absolute Error Loss; no sqrt on _testing_error
    # Number of epochs to train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for inputs, targets in dataloaders['train']:
            # Move data to the appropriate device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            model._scale1.grad = 2*(model._scale1.detach()>1)-1.
            model._scale2.grad = 2*(model._scale2.detach()>1)-1.
            model._scale3.grad = 2*(model._scale3.detach()>1)-1.
            torch.nn.utils.clip_grad_norm_([model._scale1, model._scale2, model._scale3, model._scale4, model._scale5, model._scale6, model._scale7, model._scale8, model._scale9], max_norm=1e-3)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item() * inputs.size(0)

        # Average loss for this epoch
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        model.eval()
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:1.3e}')

    torch.save(model.state_dict(), args.checkpoint)