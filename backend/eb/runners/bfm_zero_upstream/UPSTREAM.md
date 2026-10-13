# Fixed BFM-Zero network dependency

Source: [LeCAR-Lab/BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero/tree/318cf44a3262e5bdec5944f82f1a5f509b95d09b), revision `318cf44a3262e5bdec5944f82f1a5f509b95d09b`.

`base.py`, `nn_filters.py`, `nn_models.py` and `normalizers.py` are verbatim copies from `humanoidverse/agents/`. Original attribution: Meta Platforms, Inc. and affiliates; BFM-Zero authors Yitang Li, Zhengyi Luo and collaborators. License: CC BY-NC 4.0, reproduced in `LICENSE`.

Only network/filter/normalization modules are used. Training orchestration, losses, optimizer steps, replay and target updates run in the RLinf worker; the upstream trainer/agent loop is not imported.
