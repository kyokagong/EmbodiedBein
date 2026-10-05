import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero_sft import LiberoDataset, LiberoLoader, collate
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from openpi.shared.normalize import NormStats, save
root = Path(tempfile.mkdtemp())
features = {
    'observation.images.image': {'dtype':'image','shape':(32,32,3),'names':['height','width','channels']},
    'observation.images.image2': {'dtype':'image','shape':(32,32,3),'names':['height','width','channels']},
    'observation.state': {'dtype':'float32','shape':(8,), 'names':None},
    'action': {'dtype':'float32','shape':(7,), 'names':None},
}
dataset = LeRobotDataset.create('lerobot/libero',fps=10,features=features,root=root/'dataset',use_videos=False)
for i in range(12):
    dataset.add_frame({'observation.images.image':np.zeros((32,32,3),np.uint8), 'observation.images.image2':np.full((32,32,3),255,np.uint8),'observation.state':np.zeros(8,np.float32),'action':np.full(7,i,np.float32),'task':'pick up the cup'})
dataset.save_episode()
dataset.finalize()
assets = root/'model'/'physical-intelligence'/'libero'
assets.mkdir(parents=True)
save(assets, {key:NormStats(mean=np.zeros(n),std=np.ones(n),q01=np.zeros(n),q99=np.ones(n)) for key,n in [('state',8),('actions',7)]})
config=get_openpi_config('pi05_libero',model_path=str(root/'model'))
data_config=config.data.create(config.assets_dirs,config.model)
sample=LiberoDataset(str(root/'dataset'),data_config)[0]
assert sample['actions'].shape == (10,32), sample['actions'].shape
assert sample['state'].shape == (32,)
assert sample['image']['base_0_rgb'].shape == (224,224,3)
assert sample['image']['left_wrist_0_rgb'].max() == 255
assert sample['tokenized_prompt'].shape == (200,)
train_dataset=LiberoDataset(str(root/'dataset'),data_config)
def make_loader():
    sampler=DistributedSampler(train_dataset,num_replicas=1,rank=0,shuffle=True,seed=7,drop_last=True)
    return LiberoLoader(train_dataset,data_config,batch_size=2,sampler=sampler,drop_last=True,num_workers=0,collate_fn=collate)
loader=make_loader()
iterator=iter(loader)
next(iterator)
state=loader.state_dict()
expected=next(iterator)[1].clone()
restored=make_loader()
restored.load_state_dict(state)
actual=next(iter(restored))[1]
assert torch.equal(actual,expected), 'stateful loader did not restore the next batch'
assert loader.data_config() is data_config
print('PASS: LeRobot v3 transforms and stateful LIBERO loader resume the exact next batch')
