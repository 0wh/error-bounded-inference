import torch
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.parameter import Parameter
from torch.nn.init import _calculate_fan_in_and_fan_out, uniform_
import math
import argparse

from torch.utils.data import TensorDataset, DataLoader

class Model_BF(torch.nn.Module):
    def __init__(self):

        super(Model_BF, self).__init__()

        self.layer1 = torch.nn.Linear(13, 36)
        self.layer1 = spectral_norm(self.layer1)
        self.leaky_relu1 = torch.nn.LeakyReLU(negative_slope=0.1)        

        self.layer2 = torch.nn.Linear(36, 36)
        self.layer2 = spectral_norm(self.layer2)
        self.leaky_relu2 = torch.nn.LeakyReLU(negative_slope=0.1)        

        self.layer3 = torch.nn.Linear(36, 36)
        self.layer3 = spectral_norm(self.layer3)
        self.leaky_relu3 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.layer4 = torch.nn.Linear(36, 36)
        self.layer4 = spectral_norm(self.layer4)
        self.leaky_relu4 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.layer5 = torch.nn.Linear(36, 36)
        self.layer5 = spectral_norm(self.layer5)
        self.leaky_relu5 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.layer6 = torch.nn.Linear(36, 36)
        self.layer6 = spectral_norm(self.layer6)
        self.leaky_relu6 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.layer7 = torch.nn.Linear(36, 36)
        self.layer7 = spectral_norm(self.layer7)
        self.leaky_relu7 = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.layer8 = torch.nn.Linear(36, 36)
        self.layer8 = spectral_norm(self.layer8)
        self.leaky_relu8 = torch.nn.LeakyReLU(negative_slope=0.1)        
        
        self.layer9 = torch.nn.Linear(36, 3, bias=False)
        self._bias = Parameter(torch.empty(9)) #out_features
        fan_in, _ = _calculate_fan_in_and_fan_out(self.layer3.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        uniform_(self._bias, -bound, bound)
        self.layer9 = spectral_norm(self.layer9)
        u = self.layer9.state_dict()['parametrizations.weight.0._u']
        v = self.layer9.state_dict()['parametrizations.weight.0._v']
        weight_mat = self.layer9.state_dict()['parametrizations.weight.original'].flatten(1)
        sigma = torch.vdot(u, torch.mv(weight_mat, v))
        self._scale = Parameter(sigma)
    
    def forward(self, x):
        x = self.leaky_relu1(self.layer1(x))
        x = self.leaky_relu2(self.layer2(x))
        x = self.leaky_relu3(self.layer3(x))
        x = self.leaky_relu4(self.layer4(x))
        x = self.leaky_relu5(self.layer5(x))
        x = self.leaky_relu6(self.layer6(x))
        x = self.leaky_relu7(self.layer7(x))
        x = self.leaky_relu8(self.layer8(x))
        x = self._scale*self.layer9(x)+self._bias
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input tensor file")
    parser.add_argument("--target", required=True, help="Path to the target tensor file")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model_BF().to(device)

    # Create a TensorDataset and DataLoader
    input_torch_tensor = torch.load(args.input)
    target_torch_tensor = torch.load(args.target)
    dataset = TensorDataset(input_torch_tensor, target_torch_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss() # Mean Absolute Error Loss; no sqrt on _testing_error
    # Number of epochs to train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for inputs, targets in data_loader:
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
        epoch_loss = running_loss / len(data_loader.dataset)
        model.eval()
        with torch.no_grad():
            epoch_error = torch.linalg.norm(model(input_torch_tensor.to(device, torch.float32))-target_torch_tensor)/torch.sqrt(target_torch_tensor.shape[0])
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:1.3e}, Error: {epoch_error:1.3e}, Sigma: {model._scale1.item()*model._scale2.item()*model._scale3.item()*model._scale4.item()*model._scale5.item()*model._scale6.item()*model._scale7.item()*model._scale8.item()*model._scale9.item()}')

    torch.save(model.state_dict(), args.checkpoint)