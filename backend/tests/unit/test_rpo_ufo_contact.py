import numpy as np
import pytest


def test_rsi_contact_projection_filters_airborne_and_preserves_joint_state():
    from eb.runners.rpo_ufo_contact import reset_mask_and_lift

    root = np.tile([0.,0.,.7,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],(5,1)).astype(np.float32)
    q=np.zeros((5,23),np.float32);dq=q.copy()
    dq[3,19]=7
    root[4,2]=.51
    mask,lift=reset_mask_and_lift(root,q,dq,np.array([-.015,.01,.2,0.,0.]),np.zeros(5))
    assert mask.tolist()==[True,True,False,False,False]
    np.testing.assert_allclose(lift[:2],[.017,0.])
    assert np.all(q==0) and dq[3,19]==7
    with pytest.raises(ValueError):
        reset_mask_and_lift(root,q,dq,np.full(5,np.nan),np.zeros(5))


def test_contact_profile_keeps_control_rate_and_old_cfg_unchanged():
    from types import SimpleNamespace as S
    from eb.runners.rpo_ufo_contact import configure_substeps

    cfg=S(sim=S(dt=.005,render_interval=4),decimation=4)
    configure_substeps(cfg)
    assert cfg.sim.dt==.0025 and cfg.decimation==8
    assert cfg.sim.dt*cfg.decimation==.02


def test_contact_probe_refuses_forged_peak_summary(tmp_path):
    import hashlib
    from eb.runners.rpo_ufo_contact_probe import PHASES,PROTOCOL,measured_summary,validate_probe

    paired=tmp_path/'paired-inputs.npz';np.savez_compressed(paired,root=np.zeros((16,32,13)))
    files=[paired];rows=[]
    for name in PHASES:
        raw={'trace':np.zeros((16,16,32,118 if name=='baseline' else 128),np.float32),
            'terminated':np.zeros((16,16,32),bool),'active':np.ones((16,16,32),bool),
            'foot_force_N':np.ones((16,16,32),np.float32)*23,
            'peak_dq':np.ones((16,16,32),np.float32)*9,
            'boundary_dq':np.ones((16,16,32),np.float32)*7,
            'fk_before':np.zeros(16),'fk_after':np.zeros(16)}
        p=tmp_path/f'{name}.npz';np.savez_compressed(p,**raw);files.append(p)
        rows.append({'phase':name,**measured_summary(raw)})
    validation={'asset_sha256':'a'*64,'manifest_sha256':'b'*64}
    receipt={'protocol':PROTOCOL,'status':'measured',**validation,'phases':rows,
        'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    validate_probe(receipt,validation,tmp_path)
    rows[-1]['max_dq_active']=2
    with pytest.raises(ValueError,match='summary differs'):validate_probe(receipt,validation,tmp_path)
