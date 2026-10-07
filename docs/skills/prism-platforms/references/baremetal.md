# PRISM on Baremetal (rbdgx3, 8-GPU single node)

## Launcher

`tools/launch_baremetal.py` — reads `experiments/prism_designs.yaml` and
generates tuned shell scripts with `numactl` pinning + accelerate flags. No PBS;
single-node dev.

## Topology (rbdgx3, heterogeneous)

- NVLink island: GPUs 4–7 (NV6 switch).
- PCIe island: GPUs 0–3.
- 4 NUMA nodes.

Supported topology presets:

| Preset | Use | Devices |
|--------|-----|---------|
| `4gpu_nvlink` | dev | `CUDA_VISIBLE_DEVICES=4,5,6,7` |
| `8gpu_pinned` | training | all 8, with NUMA affinity |
| `1gpu_debug` | debug | `CUDA_VISIBLE_DEVICES=4` |

The launcher sets per-rank `CUDA_VISIBLE_DEVICES` + `numactl` before the python
invocation. For dev, prefer the NVLink island (4–7) to avoid the slow PCIe hop.

See [`docs/platforms/running_on_rbdgx3.md`](../../../platforms/running_on_rbdgx3.md).
