# resnet-eft

Code accompanying the paper:

**Collective Kernel EFT for Pre-activation ResNets**
Hidetoshi Kawase and Toshihiro Ota, [arXiv:2604.15742](https://arxiv.org/abs/2604.15742) (2026)

`resnet_eft` implements the collective-kernel EFT for finite-width pre-activation ResNets at initialization.

Given an input kernel $K_0^0$ and hyperparameters $(C_W, C_b, \varepsilon)$, it computes:

| Quantity | Symbol | Description |
|---|---|---|
| Mean kernel | $K_0$ | Leading-order kernel propagated through depth |
| Kernel fluctuation covariance | $V_4$ | Leading finite-width variance of $G$ |
| NLO mean-kernel correction | $K_\mathrm{eff}$ | $O(1/n)$ shift of the mean kernel |

The computation follows the Gaussian closure hierarchy (GC0 → LIN → GC1) derived in the paper.

## Installation

```bash
pip install -e ".[notebooks]"   # or: uv pip install -e ".[notebooks]"
```

Requires Python ≥ 3.11 and PyTorch ≥ 2.0.

## Reproducing Paper Figures

```bash
cd notebooks && jupyter lab
```

Open `paper_figures.ipynb` and run all cells.  
**Expected runtime**: ~8–10 hours on CPU ($M = 5\times10^6$ samples).  
For a quick test, reduce `M = 50_000` and `L = 200` in the parameters cell.

| File | Figure | Content |
|---|---|---|
| `figs/fig1_K0_V4.pdf` | Fig. 1 | $K_0$ and $V_4$ trajectories (diagonal & off-diagonal, $n=64$, $\varepsilon=0.05$) |
| `figs/fig2_V4_eps_sweep.pdf` | Fig. 2 | $V_4$ $\varepsilon$-dependence: trajectories and relative error for $\varepsilon \in \{0.10, 0.07, 0.05\}$, $n=256$ |
| `figs/fig3_K1.pdf` | Fig. 3 | $K_1$ validation: $K_\mathrm{mic}$, $K_1$(exact source), $K_\mathrm{eff}$ (EFT) trajectories |
| `figs/fig4_U1_source.pdf` | Fig. 4 | $U_1$ source: exact $U_\mathrm{ex}$ vs EFT model $U_\mathrm{mod}$ at each depth |

## Package Structure

```
src/resnet_eft/
├── core_types.py           # ActivationSpec, Params, KernelState
├── gaussian_expectation.py # Gauss-Hermite / MC expectations of σ, σ', σ''
├── layer_update.py         # resnet_step(), create_resnet_initial_state()
├── chi_op.py               # χ_K transport operator
├── k1_source_op.py         # K1SourceOp: Hess[K_0] : V_4
├── v4_repr.py              # V4Tensor, V4SliceRepr representations
├── backend.py              # PyTorch tensor utilities
└── validation/
    ├── mc_simulation.py    # Gaussian pre-activation MC (annealed)
    └── real_network.py     # Real-weight network simulation
```

### Quick API example

```python
import torch
from resnet_eft import Params, ActivationSpec, create_resnet_initial_state, resnet_step

N, n = 4, 256
K0 = torch.eye(N, dtype=torch.float64) * 2.0   # K_0^0: input kernel
params = Params(act=ActivationSpec.tanh(), Cw=2.0, Cb=0.0)
eps, L = 0.05, 400

state = create_resnet_initial_state(K0, fan_in=n, params=params)
for _ in range(L):
    state = resnet_step(state, params, eps=eps, compute_K1=True, compute_V4=True)

print(state.K0)                   # K_0 at depth L
print(state.get_physical_V4())    # V_4 / n at depth L
print(state.get_physical_K1())    # K_eff / n at depth L
```

## Citation

```bibtex
@article{kawase2026resnet,
  title={Collective Kernel {EFT} for Pre-activation {ResNets}},
  author={Kawase, Hidetoshi and Ota, Toshihiro},
  journal={arXiv preprint arXiv:2604.15742},
  year={2026}
}
```

## License

MIT
