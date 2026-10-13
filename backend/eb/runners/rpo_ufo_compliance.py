"""Frozen v11 contact, reset screening and conditional training admission."""
import math

PROFILE = 'contact_compliance_v11'
STIFFNESS = 60000.
DAMPING = 600.


def velocity_penalty(ratio):
    return (ratio-.70).clamp_min(0).square()+8*(ratio-1.).clamp_min(0).square()


def flexion_mask(q, support=None):
    import numpy as np
    if q.ndim!=2 or q.shape[1]!=23 or not np.isfinite(q).all():
        raise ValueError('Invalid RSI flexion reference')
    if support is None:support=np.ones((len(q),2),dtype=bool)
    if support.shape!=(len(q),2) or support.dtype!=bool or not support.any(-1).all():
        raise ValueError('Invalid RSI support legs')
    return (((q[:,11:13]>=.15)&(q[:,7:9]<=.10))|~support).all(-1)


def configure_material(material):
    material.compliant_contact_stiffness=STIFFNESS
    material.compliant_contact_damping=DAMPING
    material.restitution=0.


def material_receipt(stage, *, compliant=True):
    from pxr import PhysxSchema, UsdShade, UsdPhysics
    rows=[]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            api=PhysxSchema.PhysxMaterialAPI(prim)
            stiffness=api.GetCompliantContactStiffnessAttr().Get()
            if str(prim.GetPath()).startswith('/World/ground'):
                rows.append({'path':str(prim.GetPath()),'stiffness_N_per_m':float(stiffness or 0),
                    'damping_Ns_per_m':float(api.GetCompliantContactDampingAttr().Get() or 0),
                    'restitution':float(UsdPhysics.MaterialAPI(prim).GetRestitutionAttr().Get() or 0),
                    'restitution_combine':str(api.GetRestitutionCombineModeAttr().Get())})
    bindings=[]
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/ground') and prim.HasAPI(UsdPhysics.CollisionAPI):
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            bindings.append({'collision':str(prim.GetPath()),'material':str(material.GetPath())})
    selected={r['path'] for r in rows if r['stiffness_N_per_m']==(STIFFNESS if compliant else 0.)
        and r['damping_Ns_per_m']==(DAMPING if compliant else 0.) and r['restitution']==0.
        and r['restitution_combine']=='multiply'
        and all(math.isfinite(r[k]) for k in ('stiffness_N_per_m','damping_Ns_per_m','restitution'))}
    if not rows or not bindings or not all(b['material'] in selected for b in bindings):
        raise ValueError('Compliant ground material is not bound to an actual collision')
    return {'compliant_materials':rows,'ground_bindings':bindings}


def long_training_gate(records, *, start, target):
    expected=set(range(start+1,target+1))
    keys={'step','strict_shadow_length_sum','completed_episodes','physics_max_rated_ratio','first_terminated',
          'first_samples','coverage_miss','ground_outside','context_errors'}
    complete=(target-start==500 and len(records)==500
        and all(set(r)==keys and type(r['step']) is int for r in records)
        and {r['step'] for r in records}==expected
        and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for r in records for v in r.values()))
    result={'schema_version':'eb/ufo-contact-long-gate/v1','start':start,'target':target,
            'complete':complete,'passed':False}
    if not complete:return result
    windows=[]
    for low in (target-200,target-100):
        rows=[r for r in records if low<r['step']<=low+100]
        count=sum(r['completed_episodes'] for r in rows)
        windows.append(sum(r['strict_shadow_length_sum'] for r in rows)/count if count else 0.)
    samples=sum(r['first_samples'] for r in records)
    peak=max(r['physics_max_rated_ratio'] for r in records)
    rate=sum(r['first_terminated'] for r in records)/samples if samples else 1.
    errors=sum(r[k] for r in records for k in ('coverage_miss','ground_outside','context_errors'))
    result.update(strict_shadow_last_windows=windows,physics_max_rated_ratio=peak,
        first_step_termination_fraction=rate,error_count=errors,
        passed=all(v>35 for v in windows) and peak<2.5 and rate<=.001 and errors==0)
    return result


def require_long_training_gate(path, *, step):
    import hashlib,json
    file=path/'training-gate.json';records=path/'training-gate-records.json';manifest=path/'resume-manifest.json'
    if step!=524 or file.is_symlink() or not file.is_file() or file.stat().st_size>4096:
        raise ValueError('Long training requires the frozen 500-update checkpoint gate')
    metadata=json.loads(manifest.read_text());members=metadata.get('files',[])
    for member,limit in ((file,4096),(records,256*1024)):
        if member.is_symlink() or not member.is_file() or member.stat().st_size>limit:
            raise ValueError('Long training gate evidence is missing or unbounded')
        digest=hashlib.sha256(member.read_bytes()).hexdigest()
        if not any(r.get('path')==member.name and r.get('sha256')==digest for r in members):
            raise ValueError('Long training gate is not checkpoint-hash bound')
    value=json.loads(file.read_text())
    recalculated=long_training_gate(json.loads(records.read_text()),start=24,target=524)
    if recalculated!=value:
        raise ValueError('Long training gate differs from actual completed-episode evidence')
    if (metadata.get('step')!=step or value.get('start')!=24 or value.get('target')!=524
        or value.get('schema_version')!='eb/ufo-contact-long-gate/v1'
        or value.get('complete') is not True or value.get('passed') is not True
        or not all(math.isfinite(v) and v>35 for v in value.get('strict_shadow_last_windows',[]))
        or len(value.get('strict_shadow_last_windows',[]))!=2
        or not 0<=value.get('physics_max_rated_ratio',float('inf'))<2.5
        or not 0<=value.get('first_step_termination_fraction',1)<=.001
        or value.get('error_count')!=0):
        raise ValueError('Long training gate did not pass')
    return value
