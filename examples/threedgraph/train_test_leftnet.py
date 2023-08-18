import os.path as osp

import torch
from dig.threedgraph.dataset import thuEMol
from dig.threedgraph.evaluation import ThreeDEvaluator, DetailedThreeDEvaluator
from dig.threedgraph.method import LEFTNet
from dig.threedgraph.method import run
from torch_geometric.data import DataLoader
import shutil
import os


targets = ['binding_e', 'dielectric_constant', 'viscosity', 'homo', 'lumo']
target = targets[0]


data_path = r'/opt/data/0818_rdkit'
os.makedirs(data_path, exist_ok=True)
shutil.copytree(src=r'iid_test', dst=os.path.join(data_path, r'iid_test'))
shutil.copytree(src=r'ood_test', dst=os.path.join(data_path, r'ood_test'))
shutil.copytree(src=r'train', dst=os.path.join(data_path, r'train'))

model = LEFTNet(
    num_layers=6,
    hidden_channels=192,
    # hidden_channels=256,
    num_radial=96,
    # cutoff=8
    cutoff=8
)

####################################################################################################################
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

dataset = thuEMol(root=osp.join(data_path, 'train'))
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=33540, valid_size=3353, seed=42)
# split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=100, valid_size=100, seed=42)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
    split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

loss_func = torch.nn.MSELoss()
evaluation = ThreeDEvaluator()

run3d = run()
run3d.runCLR(device=device, train_dataset=train_dataset, valid_dataset=valid_dataset,
          model=model, loss_func=loss_func, evaluation=evaluation,
          batch_size=100, val_batch_size=100, epochs=2000,
          save_dir='./run_info',
          log_dir='./run_info',
          optimizer_args={'max_lr': 5e-4,
                          'base_lr': 1e-5,
                          'step_size_up': 10,
                          'step_size_down': 40,
                          'mode': "exp_range"})

# run3d.runExpo(device=device, train_dataset=train_dataset, valid_dataset=valid_dataset,
#           model=model, loss_func=loss_func, evaluation=evaluation,
#           batch_size=32, val_batch_size=100, epochs=500,
#           # batch_size=100, val_batch_size=100, epochs=250,
#           save_dir='./run_info',
#           log_dir='./run_info',
#           optimizer_args={'lr': 1e-3,
#                           'gamma': 0.987})

####################################################################################################################
ckpt = torch.load('./run_info/valid_checkpoint.pt')
model.load_state_dict(ckpt['model_state_dict'])
model.to(device=device)
####################################################################################################################
dataset = thuEMol(root=osp.join(data_path, 'ood_test'))
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1, valid_size=1, seed=42)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
    split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))
####################################################################################################################
evaluation = DetailedThreeDEvaluator(dump_info_path=r'./test_info', info_file_flag='ood_test', property=target)
info = run3d.val(model=model, data_loader=DataLoader(test_dataset, 50, shuffle=False),
                 energy_and_force=False, p=0, evaluation=evaluation, device=device)
print(f'ood_test: {info}')
####################################################################################################################
dataset = thuEMol(root=osp.join(data_path, 'iid_test'))
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1, valid_size=1, seed=42)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
    split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))
####################################################################################################################
evaluation = DetailedThreeDEvaluator(dump_info_path=r'./test_info', info_file_flag='iid_test', property=target)
info = run3d.val(model=model, data_loader=DataLoader(test_dataset, 50, shuffle=False),
                 energy_and_force=False, p=0, evaluation=evaluation, device=device)
print(f'iid_test: {info}')
####################################################################################################################
