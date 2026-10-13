"""Fixed multi-take G1 reference pool; source acquisition remains hash pinned."""
import argparse
import json
import re
from pathlib import Path

VALIDATION_TAKES={"walk3","dance2","fallAndGetUp2"}
TEST_TAKES={"walk4","run2","fightAndSports1","fallAndGetUp3"}


def select_pool(data):
    groups={}
    for name in data:
        match=re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)_subject([1-9][0-9]*)_clip([0-9]+)",name)
        if not match:
            raise ValueError("G1 pool has an unsafe/unrecognized original motion ID")
        take,subject,clip=match.groups()
        groups.setdefault(f"{take}_subject{subject}",[]).append((int(clip),name,take))
    if not 1<=len(groups)<=60:
        raise ValueError("G1 pool original motion count exceeds the bounded recipe")
    result=[]
    for group,clips in sorted(groups.items()):
        ordered=sorted(clips)
        _,name,take=next((c for c in ordered if c[0]==3),ordered[len(ordered)//2])
        split="validation" if take in VALIDATION_TAKES else "test" if take in TEST_TAKES else "train"
        result.append({"id":name,"source_group":take,"original_group":group,"split":split})
    return result


def run(source,assets,output,report,asset_sha256):
    from g1_lafan_source import load_verified,sha,SOURCES
    from g1_lafan_export import export
    from rpo_g1_r2r_trial import retarget
    data=load_verified(source)
    rows=select_pool(data)
    if output.exists():
        raise ValueError("G1 pool refuses immutable output overwrite")
    output.mkdir(parents=True,exist_ok=False)
    # Freeze selection/splits before any geometry outcome, including rejected clips.
    (output/"selection.json").write_text(json.dumps(rows,indent=2))
    clips=[]
    for row in rows:
        clip_id=row["id"]
        exported=output/"exports"/clip_id
        export(source,exported,clip_id=clip_id,frames=400,source_data=data)
        receipt_path=exported/"export-receipt.json"
        receipt=json.loads(receipt_path.read_text())
        receipt.update(source_group=row["source_group"],source_original_group=row["original_group"])
        receipt_path.write_text(json.dumps(receipt,indent=2))
        target=output/"clips"/clip_id
        target.parent.mkdir(exist_ok=True)
        retarget(exported,assets,target,output/f"{clip_id}-index.json",asset_sha256=asset_sha256,frames=400)
        mapped=json.loads((target/"retarget-receipt.json").read_text())
        clips.append({**row,"directory":f"clips/{clip_id}","retarget_sha256":sha(target/"retarget-receipt.json"),
            "frames":400,"valid_frames":mapped["valid_frames"],"continuous_9frame_starts":mapped["continuous_9frame_starts"]})
        print(json.dumps({"pool_progress":len(clips),"total":len(rows),"clip":clip_id,"valid_frames":mapped["valid_frames"]}),flush=True)
    receipt={"schema_version":"eb/rpo-g1-reference-pool/v1","status":"succeeded","clips":clips,
        "asset_sha256":asset_sha256,"source_manifest_sha256":sha(source/"source-manifest.json"),
        "selection_sha256":sha(output/"selection.json"),"source_refs":list(SOURCES.values()),
        "source_use":"local_noncommercial_no_redistribution","split_policy":"take_name_all_subjects_together",
        "frames":400*len(clips),"training_ready":False}
    (output/"pool-receipt.json").write_text(json.dumps(receipt,indent=2))
    report.write_text(json.dumps({"format":"rpo_g1_reference_pool","clips_count":len(clips),"frames_count":receipt["frames"],
        "total_bytes":sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),"source_refs":receipt["source_refs"],
        "validation":{"scope":"rpo_g1_pool_v1","status":"validated","training_ready":False,"storage_ref":str(output)}}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("source","assets","output","report"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--asset-sha256",required=True)
    a=p.parse_args()
    run(a.source,a.assets,a.output,a.report,a.asset_sha256)
