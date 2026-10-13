"""Bounded CPU pool/capacity checks; never launches training or GPU physics."""
import argparse
import json
import resource
import time
import gc
from pathlib import Path


def check_pool(output):
    from eb.runners.rpo_ufo_motion import load_robot_state
    from eb.runners.rpo_ufo_contact import ContactResetPool
    from eb.rpo_bfm import CODEC_V2_HASH
    started=time.monotonic()
    _, expert=load_robot_state(Path('/data/kk/data/ds_fb0ecda9e70644f2afdd/content'),
        asset_sha256='b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842',
        expected_codec_sha256=CODEC_V2_HASH)
    pool=ContactResetPool(expert,'/data/kk/assets/roboparty/rpo-v2',tracking_contact=True)
    result={'status':'passed','manifest_sha256':expert.manifest_sha256,
            'ordinary_candidates':{k:len(v) for k,v in pool.candidates.items()},
            'tracking_candidates':{k:len(v) for k,v in pool.tracking_candidates.items()},
            'rows':pool.rows,'wall_seconds':time.monotonic()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


def check_replay(output, capacity):
    import ctypes
    import numpy as np
    from eb.runners.bfm_ufo_replay import UFOReplayBridge
    from eb.runners.bfm_ufo_agent import OBS_DIMS,VELOCITY_AUX_NAMES
    started=time.monotonic()
    replay=UFOReplayBridge(capacity=capacity,aux_names=VELOCITY_AUX_NAMES)
    obs={key:np.zeros((1,width),np.float32) for key,width in OBS_DIMS.items()}
    replay.append({'observation':obs,'next':{'observation':obs,'terminated':np.zeros((1,1),bool),
        'truncated':np.zeros((1,1),bool)},'action':np.zeros((1,23),np.float32),
        'z':np.ones((1,256),np.float32),'aux_rewards':{key:np.zeros((1,1),np.float32) for key in VELOCITY_AUX_NAMES},
        **{key:np.zeros((1,1),np.int64) for key in ('episode_id','step_id','policy_version')}})
    for value in replay.storage.values():
        value.fill(0)
    for start in range(0,capacity,65536):
        replay.storage[('episode_id',)][start:start+65536,0]=np.arange(start,min(start+65536,capacity))
    replay.count=replay.received=capacity;replay.cursor=0
    root=output.parent/f'full-replay-{capacity}'
    state=replay.save_arrays(root)
    expected=replay.sample_numpy(16)['episode_id']
    storage_bytes=sum(v.nbytes for v in replay.storage.values())
    del replay
    gc.collect();ctypes.CDLL(None).malloc_trim(0)
    restored=UFOReplayBridge(capacity=capacity,aux_names=VELOCITY_AUX_NAMES)
    restored.load_arrays(root,state)
    np.testing.assert_array_equal(restored.sample_numpy(16)['episode_id'],expected)
    result={'status':'passed','capacity':capacity,'storage_bytes':storage_bytes,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'wall_seconds':time.monotonic()-started,'files':len(state['arrays']),
        'array_bytes':sum(p.stat().st_size for p in root.iterdir()),'next_sample_rng_match':True}
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',choices=('pool','replay'),default='pool')
    parser.add_argument('--capacity',choices=(4000000,5120000),type=int,default=4000000)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='pool':check_pool(args.output)
    else:check_replay(args.output,args.capacity)
