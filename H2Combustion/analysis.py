import torch
import numpy as np

quantBound_dict = {}
model_series_dict = {}
Lipchitz_dict = {}
innout_dict = {}

def Tol_on_raw_data(userRelativeBound, Qratio, exp='RR'):
    quantBound = quantBound_dict[exp]
    model_series = model_series_dict[exp]
    Lipchitz = Lipchitz_dict[exp]
    inputs, outputs = innout_dict[exp]
    userBound = userRelativeBound*np.linalg.norm(outputs.flatten(), ord=np.inf)
    alocQuantBound = userBound*Qratio
    quantizedModels = []
    for i, bound in enumerate(alocQuantBound):
        for q, m in zip(reversed(quantBound), model_series):
            if bound>=q:
                alocQuantBound[i] = q
                quantizedModels.append(m)
                break
    
    alocCompBound = userBound-alocQuantBound
    compBound = alocCompBound/Lipchitz
    compBoundLinf = compBound/np.sqrt(inputs.shape[1])
    compBoundLinfMDR = compBoundLinf*80
    compBoundL2 = compBound*np.sqrt(inputs.numel())/np.sqrt(inputs.shape[1])
    compBoundMSE = compBound/np.sqrt(inputs.shape[1])
    compBoundMSEMDR = compBoundMSE*2
    return quantizedModels, compBoundLinf, compBoundLinfMDR, compBoundL2, compBoundMSE, compBoundMSEMDR

def test(compBoundLinfMDR, quantizedModels, compressor='zfp'):
    achievedRelativeLinf = []
    inputs = torch.load('path_to_input_tensor_file')
    outputs = model(inputs).detach()
    for tol, model in zip(compBoundLinfMDR, quantizedModels):
        inputs_lossy = torch.load(f'path_to_reduced_data_folder/{compressor}_{tol}', dtype=torch.float32)
        outputs_lossy = model(inputs_lossy).detach()
        achievedRelativeLinf.append((torch.linalg.norm(outputs_lossy.flatten() - outputs.flatten(), ord=np.inf)/torch.linalg.norm(outputs.flatten(), ord=torch.inf)).item())
    return achievedRelativeLinf

userRelativeBound = np.logspace(-0.5, -4.5, 17) #specify a tolerance range of your interests
quantizedModels, compBoundLinf, compBoundLinfMDR, compBoundL2, compBoundMSE, compBoundMSEMDR = Tol_on_raw_data(userRelativeBound, 0.1, 'RR')

for ref in ['zfp_re', 'sz_re', 'mgard_re', 'sz_re_r2', 'mgard_re_r2']:
    achievedRelativeLinf = []
    achievedRelativeL2 = []
    if ref=='mgard_re_r2': _compBound = compBoundMSE
    elif ref=='sz_re_r2': _compBound = compBoundL2
    else: _compBound = compBoundLinf
    inputs = torch.load('path_to_input_tensor_file')
    outputs = model(inputs).detach()
    compressor = 'zfp' #valid values: 'zfp', 'sz', 'mgard'
    for tol, model in zip(_compBound, quantizedModels):
        try:
            inputs_lossy = torch.load(f'path_to_reduced_data_folder/{compressor}_{tol}', dtype=torch.float32)
        except:
            achievedRelativeLinf.append(None)
            achievedRelativeL2.append(None)
            continue
        outputs_lossy = model(inputs_lossy).detach().numpy()
    
        achievedRelativeLinf.append((np.linalg.norm(outputs_lossy.flatten() - outputs.flatten(), ord=np.inf)/np.linalg.norm(outputs.flatten(), ord=np.inf)).item())
        achievedRelativeL2.append((np.linalg.norm(outputs_lossy.flatten() - outputs.flatten())/np.linalg.norm(outputs.flatten())).item())
