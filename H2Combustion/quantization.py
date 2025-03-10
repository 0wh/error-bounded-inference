import torch
import torch.nn as nn
from train import Model_H2C
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_H2C_QAT = Model_H2C()
model_H2C_QAT.train()
model_H2C_QAT.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
model_H2C_QAT.tanh1.qconfig = None
model_H2C_QAT.tanh2.qconfig = None
torch.ao.quantization.prepare_qat(model_H2C_QAT, inplace=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint")
    parser.add_argument("--quantized", required=True, help="Path to the output folder")
    args = parser.parse_args()

    # load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    model_H2C_QAT.load_state_dict(state_dict)

    model_H2C_QAT.eval()
    model_H2C_QAT.cpu()

    model_H2C = Model_H2C(9, 9)
    model_H2C.layer1.weight.data = model_H2C_QAT.layer1.weight.data
    model_H2C.layer1.bias.data = model_H2C_QAT.layer1.bias.data
    model_H2C.layer2.weight.data = model_H2C_QAT.layer2.weight.data
    model_H2C.layer2.bias.data = model_H2C_QAT.layer2.bias.data
    model_H2C.output_layer.weight.data = model_H2C_QAT.layer3.weight.data*model_H2C_QAT._scale.data
    model_H2C.output_layer.bias.data = model_H2C_QAT._bias.data
    model_H2C.eval()

    model_H2C_fp32 = model_H2C
    model_H2C_fp16 = torch.quantization.quantize_dynamic(
        model_H2C, {nn.Linear}, dtype=torch.float16
    )
    model_H2C_int8 = torch.quantization.quantize_dynamic(
        model_H2C, {nn.Linear}, dtype=torch.qint8
    )

    #$torch.linalg.svdvals(vector)[0]==torch.linalg.norm(vector)$
    #$torch.linalg.svdvals(matrix)[0]==torch.linalg.norm(matrix, ord=2)$
    sigma3 = torch.linalg.svdvals(model_H2C.output_layer.weight.data.cpu())[0]
    sigma2 = torch.linalg.svdvals(model_H2C.layer2.weight.data.cpu())[0]
    sigma1 = torch.linalg.svdvals(model_H2C.layer1.weight.data.cpu())[0]
    quantBound_H2C = [0]
    #tf32:
    p = 2**(-10)
    q3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_H2C.output_layer.weight.data.cpu()))))))
    q2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_H2C.layer2.weight.data.cpu()))))))
    q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_H2C.layer1.weight.data.cpu()))))))
    quantBound_H2C.append((sigma2*sigma3*q1*(9*50/12)**0.5+(sigma1+q1*(9/3)**0.5)*sigma3*q2*(9*50/12)**0.5+(sigma1+q1*(9/3)**0.5)*(sigma2+q2*(50/3)**0.5)*q3*(9*9/12)**0.5).item())
    #fp16:
    p = 2**(-10)
    q3 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_H2C.output_layer.weight.data.cpu())))))))
    q2 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_H2C.layer2.weight.data.cpu())))))))
    q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.max(torch.tensor(-14.), torch.floor(torch.log2(torch.abs(model_H2C.layer1.weight.data.cpu())))))))
    quantBound_H2C.append((sigma2*sigma3*q1*(9*50/12)**0.5+(sigma1+q1*(9/3)**0.5)*sigma3*q2*(9*50/12)**0.5+(sigma1+q1*(9/3)**0.5)*(sigma2+q2*(50/3)**0.5)*q3*(9*9/12)**0.5).item())
    #bf16:
    p = 2**(-7)
    q3 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_H2C.output_layer.weight.data.cpu()))))))
    q2 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_H2C.layer2.weight.data.cpu()))))))
    q1 = p*torch.sqrt(torch.mean(torch.square(2**torch.floor(torch.log2(torch.abs(model_H2C.layer1.weight.data.cpu()))))))
    quantBound_H2C.append((sigma2*sigma3*q1*(9*50/12)**0.5+(sigma1+q1*(9/3)**0.5)*sigma3*q2*(9*50/12)**0.5+(sigma1+q1*(9/3)**0.5)*(sigma2+q2*(50/3)**0.5)*q3*(9*9/12)**0.5).item())
    #int8:
    p = 2**(-8)
    q3 = p*(torch.max(model_H2C.output_layer.weight.data.cpu())-torch.min(model_H2C.output_layer.weight.data.cpu()))
    q2 = p*(torch.max(model_H2C.layer2.weight.data.cpu())-torch.min(model_H2C.layer2.weight.data.cpu()))
    q1 = p*(torch.max(model_H2C.layer1.weight.data.cpu())-torch.min(model_H2C.layer1.weight.data.cpu()))
    quantBound_H2C.append((sigma2*sigma3*q1*(9*50/12)**0.5+(sigma1+q1*(9/3)**0.5)*sigma3*q2*(9*50/12)**0.5+(sigma1+q1*(9/3)**0.5)*(sigma2+q2*(50/3)**0.5)*q3*(9*9/12)**0.5).item())

    model_H2C_series = [model_H2C_int8, lambda x:1/9*model_H2C_int8(x)+8/9*model_H2C_fp16(x), model_H2C_fp16, model_H2C_fp16, model_H2C_fp32]
    Lipchitz_H2C = (sigma3*sigma2*sigma1).numpy()
