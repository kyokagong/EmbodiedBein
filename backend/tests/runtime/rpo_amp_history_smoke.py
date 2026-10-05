"""CPU check for AMP's per-environment frame/reset semantics."""

import torch

from eb.runners.rpo_amp_extension import AMPFrameHistory


def main():
    history = AMPFrameHistory(num_envs=2, device="cpu")
    initial = torch.zeros((2, 74))
    history.reset(torch.tensor([0, 1]), initial)
    frame1 = torch.ones((2, 74))
    window, valid = history.advance(frame1)
    assert window.shape == (2, 3, 74)
    assert not valid.any()
    frame2 = torch.full((2, 74), 2.0)
    window, valid = history.advance(frame2)
    assert valid.tolist() == [True, True]
    assert window[0, :, 0].tolist() == [0, 1, 2]
    history.reset(torch.tensor([0]), torch.full((1, 74), 9.0))
    window, valid = history.advance(torch.tensor([[10.0] * 74, [3.0] * 74]))
    assert valid.tolist() == [False, True]
    assert window[1, :, 0].tolist() == [1, 2, 3]
    print("AMP frame history: pass")


if __name__ == "__main__":
    main()
