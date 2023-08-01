import torch
YOUR_CKPT_PATH = None
file_path = YOUR_CKPT_PATH
model = torch.load(file_path, map_location='cpu')
all = 0
for key in list(model['state_dict'].keys()):
    all += model['state_dict'][key].nelement()
print(all)
