"""CPU check for AMP reward, LSGAN loss and real discriminator updates."""

import torch

from eb.runners.rpo_amp_worker import AMPDiscriminator, discriminator_loss, style_reward


def main():
    assert torch.allclose(
        style_reward(torch.tensor([-1.0, 0.0, 1.0, 3.0]),
                     torch.ones(4, dtype=torch.bool), 1.5),
        torch.tensor([0.0, 0.0225, 0.03, 0.0]),
    )
    torch.manual_seed(7)
    model = AMPDiscriminator()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    expert = torch.randn((8, 222), requires_grad=True)
    policy = torch.randn((8, 222)) + 0.5
    before = model.net[0].weight.detach().clone()
    loss = discriminator_loss(model, expert, policy, grad_penalty_scale=10.0)
    optim.zero_grad()
    loss.backward()
    optim.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.net[0].weight)
    print("AMP discriminator: pass")


if __name__ == "__main__":
    main()
