import torch, torchvision
import numpy as np
import tifffile
from torch.profiler import profile, record_function, ProfilerActivity

quantBound_dict = {}
model_series_dict = {}
Lipchitz_dict = {}
innout_dict = {}

def Tol_on_raw_data(userRelativeBound, Qratio, exp='H2C'):
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
quantizedModels, compBoundLinf, compBoundLinfMDR, compBoundL2, compBoundMSE, compBoundMSEMDR = Tol_on_raw_data(userRelativeBound, 0.1, 'H2C')

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

# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.Normalize([0.0144, 0.0159, 0.0170], [0.0052, 0.0036, 0.0031]),
])
def tiff_loader(path):
    image = tifffile.imread(path)
    # EuroSAT bands are 1-indexed, PyTorch tensors are 0-indexed
    # Band 4 (Red), Band 3 (Green), Band 2 (Blue)
    image = image[:, :, (3, 2, 1)].astype(np.float32)
    image = torch.from_numpy(image/65535).permute(2,0,1)
    return image

data_dir = 'path_to_EuroSAT_dataset'
image_dataset = torchvision.datasets.ImageFolder(data_dir, data_transforms, loader=tiff_loader)

image_dataset, _ = torch.utils.data.random_split(image_dataset, [1024, 25976], generator=torch.Generator().manual_seed(42))

data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1024)


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

module = torchvision.models.resnet18
model = module(weights='DEFAULT').to(device)
model.eval()

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_stack=True) as prof:
        with record_function("{inference}"):
            for inputs, _ in data_loader:
                with record_function("{tensor_converting}"):
                    inputs = inputs.to(device)
    
                with record_function("{model_execution}"):
                    outputs = model(inputs)

del inputs, outputs, model

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=1))
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=1))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=3))
prof.export_chrome_trace("trace_{}.json".format(module.__name__))

stats = torch.cuda.memory_stats()
print('CUDA Mem {:.1f}%'.format(stats["allocated_bytes.all.peak"]/1024**2/13615*100))
