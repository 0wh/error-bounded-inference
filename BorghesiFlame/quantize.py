import torch
import torch.nn as nn
from train import Model_BF

model_BF = Model_BF()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load checkpoint
state_dict = torch.load('path_to_model_checkpoint', map_location=device)
model_BF.load_state_dict(state_dict)

model_BF.eval()

model_BF_fp32 = model_BF
model_BF_fp16 = torch.quantization.quantize_dynamic(
    model_BF, {nn.Linear}, dtype=torch.float16
)
model_BF_int8 = torch.quantization.quantize_dynamic(
    model_BF, {nn.Linear}, dtype=torch.qint8
)

#torch.linalg.svdvals(vector)[0] is equivalent to torch.linalg.norm(vector)
#torch.linalg.svdvals(matrix)[0] is equivalent to torch.linalg.norm(matrix, ord=2)
sigma9 = torch.linalg.svdvals(model_BF.output_layer.weight.data.cpu())[0]
sigma8 = torch.linalg.svdvals(model_BF.layer8.weight.data.cpu())[0]
sigma7 = torch.linalg.svdvals(model_BF.layer7.weight.data.cpu())[0]
sigma6 = torch.linalg.svdvals(model_BF.layer6.weight.data.cpu())[0]
sigma5 = torch.linalg.svdvals(model_BF.layer5.weight.data.cpu())[0]
sigma4 = torch.linalg.svdvals(model_BF.layer4.weight.data.cpu())[0]
sigma3 = torch.linalg.svdvals(model_BF.layer3.weight.data.cpu())[0]
sigma2 = torch.linalg.svdvals(model_BF.layer2.weight.data.cpu())[0]
sigma1 = torch.linalg.svdvals(model_BF.layer1.weight.data.cpu())[0]
quantBound_BF = [0]
#tf32:
p = 2**(-10)
q9 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.output_layer.weight.data.cpu()))))))
q8 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer8.weight.data.cpu()))))))
q7 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer7.weight.data.cpu()))))))
q6 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer6.weight.data.cpu()))))))
q5 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer5.weight.data.cpu()))))))
q4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer4.weight.data.cpu()))))))
q3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer3.weight.data.cpu()))))))
q2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer2.weight.data.cpu()))))))
q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer1.weight.data.cpu()))))))
quantBound_BF.append((
                                                                                                                             sigma2*sigma3*sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q1*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*                                                                                                         sigma3*sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q2*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*                                                                                          sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q3*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*                                                                           sigma5*sigma6*sigma7*sigma8*sigma9*q4*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*                                                            sigma6*sigma7*sigma8*sigma9*q5*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*                                             sigma7*sigma8*sigma9*q6*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)                              *sigma8*sigma9*q7*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)*(sigma7+q7*(36/3)**0.5)               *sigma9*q8*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)*(sigma7+q7*(36/3)**0.5)*(sigma8+q8*(36/3)**0.5)*q9*(13*3/12)**0.5
).item())
#fp16:
p = 2**(-10)
q9 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.output_layer.weight.data.cpu())))))))
q8 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.layer8.weight.data.cpu())))))))
q7 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.layer7.weight.data.cpu())))))))
q6 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.layer6.weight.data.cpu())))))))
q5 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.layer5.weight.data.cpu())))))))
q4 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.layer4.weight.data.cpu())))))))
q3 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.layer3.weight.data.cpu())))))))
q2 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.layer2.weight.data.cpu())))))))
q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_BF.layer1.weight.data.cpu())))))))
quantBound_BF.append((
                                                                                                                             sigma2*sigma3*sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q1*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*                                                                                                         sigma3*sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q2*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*                                                                                          sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q3*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*                                                                           sigma5*sigma6*sigma7*sigma8*sigma9*q4*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*                                                            sigma6*sigma7*sigma8*sigma9*q5*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*                                             sigma7*sigma8*sigma9*q6*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)                              *sigma8*sigma9*q7*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)*(sigma7+q7*(36/3)**0.5)               *sigma9*q8*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)*(sigma7+q7*(36/3)**0.5)*(sigma8+q8*(36/3)**0.5)*q9*(13*3/12)**0.5
).item())#bf16:
p = 2**(-7)
q9 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.output_layer.weight.data.cpu()))))))
q8 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer8.weight.data.cpu()))))))
q7 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer7.weight.data.cpu()))))))
q6 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer6.weight.data.cpu()))))))
q5 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer5.weight.data.cpu()))))))
q4 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer4.weight.data.cpu()))))))
q3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer3.weight.data.cpu()))))))
q2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer2.weight.data.cpu()))))))
q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_BF.layer1.weight.data.cpu()))))))
quantBound_BF.append((
                                                                                                                             sigma2*sigma3*sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q1*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*                                                                                                         sigma3*sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q2*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*                                                                                          sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q3*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*                                                                           sigma5*sigma6*sigma7*sigma8*sigma9*q4*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*                                                            sigma6*sigma7*sigma8*sigma9*q5*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*                                             sigma7*sigma8*sigma9*q6*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)                              *sigma8*sigma9*q7*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)*(sigma7+q7*(36/3)**0.5)               *sigma9*q8*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)*(sigma7+q7*(36/3)**0.5)*(sigma8+q8*(36/3)**0.5)*q9*(13*3/12)**0.5
).item())
#int8:
p = 2**(-8)
q9 = p*(torch.max(model_BF.output_layer.weight.data.cpu())-torch.min(model_BF.output_layer.weight.data.cpu()))
q8 = p*(torch.max(model_BF.layer8.weight.data.cpu())-torch.min(model_BF.layer8.weight.data.cpu()))
q7 = p*(torch.max(model_BF.layer7.weight.data.cpu())-torch.min(model_BF.layer7.weight.data.cpu()))
q6 = p*(torch.max(model_BF.layer6.weight.data.cpu())-torch.min(model_BF.layer6.weight.data.cpu()))
q5 = p*(torch.max(model_BF.layer5.weight.data.cpu())-torch.min(model_BF.layer5.weight.data.cpu()))
q4 = p*(torch.max(model_BF.layer4.weight.data.cpu())-torch.min(model_BF.layer4.weight.data.cpu()))
q3 = p*(torch.max(model_BF.layer3.weight.data.cpu())-torch.min(model_BF.layer3.weight.data.cpu()))
q2 = p*(torch.max(model_BF.layer2.weight.data.cpu())-torch.min(model_BF.layer2.weight.data.cpu()))
q1 = p*(torch.max(model_BF.layer1.weight.data.cpu())-torch.min(model_BF.layer1.weight.data.cpu()))
quantBound_BF.append((
                                                                                                                             sigma2*sigma3*sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q1*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*                                                                                                         sigma3*sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q2*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*                                                                                          sigma4*sigma5*sigma6*sigma7*sigma8*sigma9*q3*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*                                                                           sigma5*sigma6*sigma7*sigma8*sigma9*q4*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*                                                            sigma6*sigma7*sigma8*sigma9*q5*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*                                             sigma7*sigma8*sigma9*q6*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)                              *sigma8*sigma9*q7*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)*(sigma7+q7*(36/3)**0.5)               *sigma9*q8*(13*36/12)**0.5
    +(sigma1+q1*(13/3)**0.5)*(sigma2+q2*(36/3)**0.5)*(sigma3+q3*(36/3)**0.5)*(sigma4+q4*(36/3)**0.5)*(sigma5+q5*(36/3)**0.5)*(sigma6+q6*(36/3)**0.5)*(sigma7+q7*(36/3)**0.5)*(sigma8+q8*(36/3)**0.5)*q9*(13*3/12)**0.5
).item())

model_BF_series = [model_BF_int8, lambda x:1/9*model_BF_int8(x)+8/9*model_BF_fp16(x), model_BF_fp16, model_BF_fp16, model_BF_fp32]
Lipchitz_BF = (sigma9*sigma8*sigma7*sigma6*sigma5*sigma4*sigma3*sigma2*sigma1).numpy()