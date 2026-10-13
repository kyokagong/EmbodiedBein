"""Episode-local UFO tracking: other environments never own this cursor."""
import numpy as np
import torch


PROFILE = "environment_tracking_v10"


def tracking_mask(num_envs, seed):
    if num_envs < 2 or num_envs % 2:
        raise ValueError("Tracking partition requires an even environment count")
    result = np.zeros(num_envs, dtype=bool)
    result[np.random.default_rng(seed).permutation(num_envs)[:num_envs // 2]] = True
    return result


class EnvironmentRolloutContext:
    def __init__(self, num_envs, *, seed, device):
        self.mask = torch.as_tensor(tracking_mask(num_envs, seed), device=device)
        self.episode_ids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.starts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.references = torch.full((num_envs, 3), -1, dtype=torch.long, device=device)
        # Allocate cache only for the tracking half, never for all 2048 environments.
        self.tracking_ids = self.mask.nonzero().flatten()
        self.cache_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.cache_index[self.tracking_ids] = torch.arange(len(self.tracking_ids), device=device)
        self.tracking_z = torch.zeros(len(self.tracking_ids), 250, 256, device=device)
        self.tracking_q = torch.zeros(len(self.tracking_ids), 250, 23, device=device)
        self.encoding_versions = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.z = torch.zeros(num_envs, 256, device=device)
        self.last_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, native, expert, episode_ids, steps, references, version):
        device = self.z.device
        episodes, steps = episode_ids.reshape(-1).to(device), steps.reshape(-1).to(device)
        references = references.to(device)
        reset = episodes != self.episode_ids
        if ((steps < self.last_steps) & ~reset).any():
            raise ValueError("Tracking cursor moved backward within episode")
        mixed = ~self.mask & (reset | ((steps % 100 == 0) & (steps != self.last_steps)))
        if mixed.any():
            count = int(mixed.sum())
            self.z[mixed] = (native.z_buffer.sample(count, device=device) if not native.z_buffer.empty()
                             else native._model.sample_z(count, device=device))
        renew = self.mask & (reset | (steps - self.starts >= 250))
        renew_ids=renew.nonzero().flatten().tolist()
        for offset in range(0,len(renew_ids),16):
            ids=renew_ids[offset:offset+16]
            batches=[]; q_batches=[]
            for env_id in ids:
                if reset[env_id]:
                    source,clip,frame=map(int,references[env_id].tolist())
                    if (source not in (0,1) or not 0<=clip<len(expert.clips)
                            or expert.clips[clip]["source"] != ("official_rpo","g1")[source]):
                        raise ValueError("Tracking RSI reference source differs")
                    batch=expert.sample_at(clip,frame,seq_length=250)
                    self.references[env_id]=references[env_id]
                else:
                    batch=expert.sample(250,seq_length=250)
                    clip,frame=map(int,batch["sample_ids"][0].tolist())
                    source=("official_rpo","g1").index(expert.clips[clip]["source"])
                    self.references[env_id]=self.references.new_tensor([source,clip,frame])
                batches.append(batch["next"]["observation"])
                q_batches.append(batch["observation"]["state"][:,:23])
            obs={key:torch.cat([batch[key] for batch in batches]) for key in batches[0]}
            encoded=native._model.backward_map(obs).reshape(len(ids),250,256)
            cumulative=torch.cat((torch.zeros_like(encoded[:,:1]),encoded.cumsum(1)),1)
            times=torch.arange(250,device=device); ends=(times+8).clamp_max(250)
            averaged=(cumulative[:,ends]-cumulative[:,times])/(ends-times)[None,:,None]
            z=native._model.project_z(averaged.reshape(-1,256)).reshape(len(ids),250,256)
            self.tracking_z[self.cache_index[ids]]=z
            self.tracking_q[self.cache_index[ids]]=torch.stack(q_batches)
            self.starts[ids]=steps[ids]; self.encoding_versions[ids]=version
        self.z[self.tracking_ids] = self.tracking_z[
            self.cache_index[self.tracking_ids], steps[self.tracking_ids] - self.starts[self.tracking_ids]]
        self.episode_ids.copy_(episodes)
        self.last_steps.copy_(steps)
        return self.z.clone()

    def state_dict(self):
        return {"schema_version": "eb/ufo-environment-context/v1", **{
            key: getattr(self, key).detach().cpu().clone() for key in
            ("mask", "episode_ids", "starts", "references", "tracking_z", "tracking_q", "encoding_versions", "z", "last_steps")}}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected) or state["schema_version"] != expected["schema_version"]:
            raise ValueError("Missing environment tracking context state")
        if not torch.equal(state["mask"].cpu(), self.mask.cpu()):
            raise ValueError("Tracking partition changed")
        for key in expected.keys() - {"schema_version", "mask"}:
            value = state[key]
            target = getattr(self, key)
            if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
                raise ValueError("Malformed environment tracking context")
            target.copy_(value.to(target.device))
