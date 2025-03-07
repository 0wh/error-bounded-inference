import torch, torchvision
import torch.nn as nn
from train import dataloaders

model_ESAT = torchvision.models.resnet18()
num_ftrs = model_ESAT.fc.in_features
model_ESAT.fc = nn.Linear(num_ftrs, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load checkpoint
state_dict = torch.load('path_to_model_checkpoint', map_location=device)
model_ESAT.load_state_dict(state_dict)

def deepcopy(model_ESAT):
    model_copy= torchvision.models.resnet18().to(device)
    num_ftrs = model_copy.fc.in_features
    model_copy.fc = nn.Linear(num_ftrs, 10)
    state_dict = torch.load(model_ESAT, map_location=device)
    model_copy.load_state_dict(state_dict)
    model_copy.eval()
    return model_copy

model_ESAT.eval()

model_ESAT_fp32 = model_ESAT
model_ESAT_fp16 = deepcopy(model_ESAT).half()
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)
# capture_pre_autograd_graph is a short term API, it will be updated to use the offical torch.export API when that is ready.

inputs_ESAT, _ = next(iter(dataloaders['train']))

example_inputs = (inputs_ESAT,)
exported_model = capture_pre_autograd_graph(deepcopy(model_ESAT), example_inputs)

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
prepared_model = prepare_pt2e(exported_model, quantizer)
# calibrate
prepared_model(inputs_ESAT)
# run calibration on sample data
model_ESAT_int8 = convert_pt2e(prepared_model)

sigma21 = 1#_sigma(model_ESAT.fc) # average pooling
sigma4_4 = 1#_sigma(model_ESAT.layer4[1].conv2)*_sigma(model_ESAT.layer4[1].bn2)
sigma4_3 = 1#_sigma(model_ESAT.layer4[1].conv1)*_sigma(model_ESAT.layer4[1].bn1)
sigma4_0 = 1#_sigma(model_ESAT.layer4[0].downsample[0])*_sigma(model_ESAT.layer4[0].downsample[1])
sigma4_2 = 1#_sigma(model_ESAT.layer4[0].conv2)*_sigma(model_ESAT.layer4[0].bn2)
sigma4_1 = 1#_sigma(model_ESAT.layer4[0].conv1)*_sigma(model_ESAT.layer4[0].bn1)
sigma3_4 = 1#_sigma(model_ESAT.layer3[1].conv2)*_sigma(model_ESAT.layer3[1].bn2)
sigma3_3 = 1#_sigma(model_ESAT.layer3[1].conv1)*_sigma(model_ESAT.layer3[1].bn1)
sigma3_0 = 1#_sigma(model_ESAT.layer3[0].downsample[0])*_sigma(model_ESAT.layer3[0].downsample[1])
sigma3_2 = 1#_sigma(model_ESAT.layer3[0].conv2)*_sigma(model_ESAT.layer3[0].bn2)
sigma3_1 = 1#_sigma(model_ESAT.layer3[0].conv1)*_sigma(model_ESAT.layer3[0].bn1)
sigma2_4 = 1#_sigma(model_ESAT.layer2[1].conv2)*_sigma(model_ESAT.layer2[1].bn2)
sigma2_3 = 1#_sigma(model_ESAT.layer2[1].conv1)*_sigma(model_ESAT.layer2[1].bn1)
sigma2_0 = 1#_sigma(model_ESAT.layer2[0].downsample[0])*_sigma(model_ESAT.layer2[0].downsample[1])
sigma2_2 = 1#_sigma(model_ESAT.layer2[0].conv2)*_sigma(model_ESAT.layer2[0].bn2)
sigma2_1 = 1#_sigma(model_ESAT.layer2[0].conv1)*_sigma(model_ESAT.layer2[0].bn1)
sigma1_4 = 1#_sigma(model_ESAT.layer1[1].conv2)*_sigma(model_ESAT.layer1[1].bn2)
sigma1_3 = 1#_sigma(model_ESAT.layer1[1].conv1)*_sigma(model_ESAT.layer1[1].bn1)
sigma1_2 = 1#_sigma(model_ESAT.layer1[0].conv2)*_sigma(model_ESAT.layer1[0].bn2)
sigma1_1 = 1#_sigma(model_ESAT.layer1[0].conv1)*_sigma(model_ESAT.layer1[0].bn1)
sigma1 = 1/30#_sigma(model_ESAT.conv1)*_sigma(model_ESAT.bn1)
quantBound_ESAT = [0]

#tf32:
p = 2**(-10)
q21 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.fc.weight.data.cpu()))))))
q4_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[1].conv2.weight.data.cpu()))))))
q4_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[1].conv1.weight.data.cpu()))))))
q4_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].downsample[0].weight.data.cpu()))))))
q4_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].conv2.weight.data.cpu()))))))
q4_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].conv1.weight.data.cpu()))))))
q3_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[1].conv2.weight.data.cpu()))))))
q3_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[1].conv1.weight.data.cpu()))))))
q3_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].downsample[0].weight.data.cpu()))))))
q3_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].conv2.weight.data.cpu()))))))
q3_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].conv1.weight.data.cpu()))))))
q2_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[1].conv2.weight.data.cpu()))))))
q2_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[1].conv1.weight.data.cpu()))))))
q2_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].downsample[0].weight.data.cpu()))))))
q2_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].conv2.weight.data.cpu()))))))
q2_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].conv1.weight.data.cpu()))))))
q1_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer1[1].conv2.weight.data.cpu()))))))
q1_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer1[1].conv1.weight.data.cpu()))))))
q1_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer1[0].conv2.weight.data.cpu()))))))
q1_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer1[0].conv1.weight.data.cpu()))))))
q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.conv1.weight.data.cpu())))))) 
quantBound_ESAT.append((
    sigma1_1*sigma1_2*sigma1_3*sigma1_4*q1*(224*224*112*112/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*sigma1_2*sigma1_3*sigma1_4*q1_1*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*sigma1_3*sigma1_4*q1_2*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*(sigma1_2+q1_2*(56*56/3)**0.5)*sigma1_4*q1_3*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*(sigma1_2+q1_2*(56*56/3)**0.5)*(sigma1_3+q1_3*(56*56/3)**0.5)*q1_4*(224*224*56*56/12)**0.5
    #+q1*(224*224*112*112/12)**0.5 # skip connection
).item())

#fp16:
p = 2**(-10)
q21 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.fc.weight.data.cpu())))))))
q4_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer4[1].conv2.weight.data.cpu())))))))
q4_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer4[1].conv1.weight.data.cpu())))))))
q4_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].downsample[0].weight.data.cpu())))))))
q4_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].conv2.weight.data.cpu())))))))
q4_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].conv1.weight.data.cpu())))))))
q3_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer3[1].conv2.weight.data.cpu())))))))
q3_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer3[1].conv1.weight.data.cpu())))))))
q3_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].downsample[0].weight.data.cpu())))))))
q3_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].conv2.weight.data.cpu())))))))
q3_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].conv1.weight.data.cpu())))))))
q2_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer2[1].conv2.weight.data.cpu())))))))
q2_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer2[1].conv1.weight.data.cpu())))))))
q2_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].downsample[0].weight.data.cpu())))))))
q2_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].conv2.weight.data.cpu())))))))
q2_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].conv1.weight.data.cpu())))))))
q1_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer1[1].conv2.weight.data.cpu())))))))
q1_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer1[1].conv1.weight.data.cpu())))))))
q1_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer1[0].conv2.weight.data.cpu())))))))
q1_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.layer1[0].conv1.weight.data.cpu())))))))
q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_ESAT.conv1.weight.data.cpu())))))))
quantBound_ESAT.append((
    sigma1_1*sigma1_2*sigma1_3*sigma1_4*q1*(224*224*112*112/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*sigma1_2*sigma1_3*sigma1_4*q1_1*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*sigma1_3*sigma1_4*q1_2*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*(sigma1_2+q1_2*(56*56/3)**0.5)*sigma1_4*q1_3*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*(sigma1_2+q1_2*(56*56/3)**0.5)*(sigma1_3+q1_3*(56*56/3)**0.5)*q1_4*(224*224*56*56/12)**0.5
    #+q1*(224*224*112*112/12)**0.5 # skip connection
).item())

#bf16:
p = 2**(-7)
q21 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.fc.weight.data.cpu()))))))
q4_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[1].conv2.weight.data.cpu()))))))
q4_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[1].conv1.weight.data.cpu()))))))
q4_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].downsample[0].weight.data.cpu()))))))
q4_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].conv2.weight.data.cpu()))))))
q4_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer4[0].conv1.weight.data.cpu()))))))
q3_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[1].conv2.weight.data.cpu()))))))
q3_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[1].conv1.weight.data.cpu()))))))
q3_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].downsample[0].weight.data.cpu()))))))
q3_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].conv2.weight.data.cpu()))))))
q3_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer3[0].conv1.weight.data.cpu()))))))
q2_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[1].conv2.weight.data.cpu()))))))
q2_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[1].conv1.weight.data.cpu()))))))
q2_0 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].downsample[0].weight.data.cpu()))))))
q2_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].conv2.weight.data.cpu()))))))
q2_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer2[0].conv1.weight.data.cpu()))))))
q1_4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer1[1].conv2.weight.data.cpu()))))))
q1_3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer1[1].conv1.weight.data.cpu()))))))
q1_2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer1[0].conv2.weight.data.cpu()))))))
q1_1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.layer1[0].conv1.weight.data.cpu()))))))
q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_ESAT.conv1.weight.data.cpu())))))) 
quantBound_ESAT.append((
    sigma1_1*sigma1_2*sigma1_3*sigma1_4*q1*(224*224*112*112/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*sigma1_2*sigma1_3*sigma1_4*q1_1*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*sigma1_3*sigma1_4*q1_2*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*(sigma1_2+q1_2*(56*56/3)**0.5)*sigma1_4*q1_3*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*(sigma1_2+q1_2*(56*56/3)**0.5)*(sigma1_3+q1_3*(56*56/3)**0.5)*q1_4*(224*224*56*56/12)**0.5
    #+q1*(224*224*112*112/12)**0.5 # skip connection
).item())

#int8:
p = 2**(-8)
q21 = p*(torch.max(model_ESAT.fc.weight.data.cpu())-torch.min(model_ESAT.fc.weight.data.cpu()))
q4_4 = p*(torch.max(model_ESAT.layer4[1].conv2.weight.data.cpu())-torch.min(model_ESAT.layer4[1].conv2.weight.data.cpu()))
q4_3 = p*(torch.max(model_ESAT.layer4[1].conv1.weight.data.cpu())-torch.min(model_ESAT.layer4[1].conv1.weight.data.cpu()))
q4_0 = p*(torch.max(model_ESAT.layer4[0].downsample[0].weight.data.cpu())-torch.min(model_ESAT.layer4[0].downsample[0].weight.data.cpu()))
q4_2 = p*(torch.max(model_ESAT.layer4[0].conv2.weight.data.cpu())-torch.min(model_ESAT.layer4[0].conv2.weight.data.cpu()))
q4_1 = p*(torch.max(model_ESAT.layer4[0].conv1.weight.data.cpu())-torch.min(model_ESAT.layer4[0].conv1.weight.data.cpu()))
q3_4 = p*(torch.max(model_ESAT.layer3[1].conv2.weight.data.cpu())-torch.min(model_ESAT.layer3[1].conv2.weight.data.cpu()))
q3_3 = p*(torch.max(model_ESAT.layer3[1].conv1.weight.data.cpu())-torch.min(model_ESAT.layer3[1].conv1.weight.data.cpu()))
q3_0 = p*(torch.max(model_ESAT.layer3[0].downsample[0].weight.data.cpu())-torch.min(model_ESAT.layer3[0].downsample[0].weight.data.cpu()))
q3_2 = p*(torch.max(model_ESAT.layer3[0].conv2.weight.data.cpu())-torch.min(model_ESAT.layer3[0].conv2.weight.data.cpu()))
q3_1 = p*(torch.max(model_ESAT.layer3[0].conv1.weight.data.cpu())-torch.min(model_ESAT.layer3[0].conv1.weight.data.cpu()))
q2_4 = p*(torch.max(model_ESAT.layer2[1].conv2.weight.data.cpu())-torch.min(model_ESAT.layer2[1].conv2.weight.data.cpu()))
q2_3 = p*(torch.max(model_ESAT.layer2[1].conv1.weight.data.cpu())-torch.min(model_ESAT.layer2[1].conv1.weight.data.cpu()))
q2_0 = p*(torch.max(model_ESAT.layer2[0].downsample[0].weight.data.cpu())-torch.min(model_ESAT.layer2[0].downsample[0].weight.data.cpu()))
q2_2 = p*(torch.max(model_ESAT.layer2[0].conv2.weight.data.cpu())-torch.min(model_ESAT.layer2[0].conv2.weight.data.cpu()))
q2_1 = p*(torch.max(model_ESAT.layer2[0].conv1.weight.data.cpu())-torch.min(model_ESAT.layer2[0].conv1.weight.data.cpu()))
q1_4 = p*(torch.max(model_ESAT.layer1[1].conv2.weight.data.cpu())-torch.min(model_ESAT.layer1[1].conv2.weight.data.cpu()))
q1_3 = p*(torch.max(model_ESAT.layer1[1].conv1.weight.data.cpu())-torch.min(model_ESAT.layer1[1].conv1.weight.data.cpu()))
q1_2 = p*(torch.max(model_ESAT.layer1[0].conv2.weight.data.cpu())-torch.min(model_ESAT.layer1[0].conv2.weight.data.cpu()))
q1_1 = p*(torch.max(model_ESAT.layer1[0].conv1.weight.data.cpu())-torch.min(model_ESAT.layer1[0].conv1.weight.data.cpu()))
q1 = p*(torch.max(model_ESAT.conv1.weight.data.cpu())-torch.min(model_ESAT.conv1.weight.data.cpu()))
quantBound_ESAT.append((
    sigma1_1*sigma1_2*sigma1_3*sigma1_4*q1*(224*224*112*112/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*sigma1_2*sigma1_3*sigma1_4*q1_1*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*sigma1_3*sigma1_4*q1_2*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*(sigma1_2+q1_2*(56*56/3)**0.5)*sigma1_4*q1_3*(224*224*56*56/12)**0.5
    +(sigma1+q1*(112*112/3)**0.5)*(sigma1_1+q1_1*(56*56/3)**0.5)*(sigma1_2+q1_2*(56*56/3)**0.5)*(sigma1_3+q1_3*(56*56/3)**0.5)*q1_4*(224*224*56*56/12)**0.5
    #+q1*(224*224*112*112/12)**0.5 # skip connection
).item())

model_ESAT_series = [model_ESAT_int8, lambda x:1/9*model_ESAT_int8(x)+8/9*model_ESAT_fp16(x.half()), lambda x:model_ESAT_fp16(x.half()), lambda x:model_ESAT_fp16(x.half()), model_ESAT_fp32]
Lipchitz_ESAT = (sigma1*2**8)