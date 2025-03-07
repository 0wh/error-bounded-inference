import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.parameter import Parameter
from torch.nn.init import _calculate_fan_in_and_fan_out, uniform_
import math

from torch.utils.data import TensorDataset, DataLoader

class Model_H2C(torch.nn.Module):
    def __init__(self):
        
        super(Model_H2C, self).__init__()
        
        self.layer1 = torch.nn.Linear(9, 50)
        self.layer1 = spectral_norm(self.layer1)
        self.tanh1 = torch.nn.Tanh()
        
        self.layer2 = torch.nn.Linear(50, 50)
        self.layer2 = spectral_norm(self.layer2)
        self.tanh2 = torch.nn.Tanh()
        
        self.layer3 = torch.nn.Linear(50, 9, bias=False)
        self._bias = Parameter(torch.empty(9)) #out_features
        fan_in, _ = _calculate_fan_in_and_fan_out(self.layer3.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        uniform_(self._bias, -bound, bound)
        self.layer3 = spectral_norm(self.layer3)
        u = self.layer3.state_dict()['parametrizations.weight.0._u']
        v = self.layer3.state_dict()['parametrizations.weight.0._v']
        weight_mat = self.layer3.state_dict()['parametrizations.weight.original'].flatten(1)
        sigma = torch.vdot(u, torch.mv(weight_mat, v))
        self._scale = Parameter(sigma)
        
    def forward(self, x):
        x = self.tanh1(self.layer1(x))
        x = self.tanh2(self.layer2(x))
        x = self._scale*self.layer3(x)+self._bias
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model_H2C().to(device)

# Create a TensorDataset and DataLoader
input_torch_tensor = torch.load('path_to_input_tensor_file')
target_torch_tensor = torch.load('path_to_target_tensor_file')
dataset = TensorDataset(input_torch_tensor, target_torch_tensor)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # Mean Squared Error Loss
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
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item() * inputs.size(0)

    # Average loss for this epoch
    epoch_loss = running_loss / len(data_loader.dataset)
    model.eval()
    with torch.no_grad():
        epoch_error = torch.linalg.norm(model(input_torch_tensor.to(device, torch.float32))-target_torch_tensor)/torch.sqrt(target_torch_tensor.shape[0])
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss:1.3e}, Error: {epoch_error:1.3e}')

torch.save(model.state_dict(), 'path_to_model_checkpoint')
