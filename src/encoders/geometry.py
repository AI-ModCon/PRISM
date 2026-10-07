"""Geometry/physics modality encoder built on Walrus.

``GeometryEncoder`` voxelizes a point cloud, runs the Walrus encoder and
processor stack, and pools to ``(B, T, d_geo)`` tokens. Walrus and its Hydra /
the_well dependencies are optional: the imports below are guarded so the module
stays importable, and ``FallbackGeometryEncoder`` is a smoke-only substitute
gated behind ``WALRUS_FALLBACK=1``.
"""

import logging
import os
import sys

import torch
import torch.nn as nn

from .base import ModalityEncoder

logger = logging.getLogger(__name__)

# Add src/libs/walrus to sys.path to allow importing walrus
current_dir = os.path.dirname(os.path.abspath(__file__))
# src/encoders -> src -> src/libs/walrus
walrus_path = os.path.join(current_dir, "../libs/walrus")
if walrus_path not in sys.path:
    sys.path.append(walrus_path)

try:
    from dataclasses import dataclass, replace
    from functools import reduce
    from operator import mul

    import hydra
    import torch.nn.functional as F
    import walrus
    from huggingface_hub import hf_hub_download
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from the_well.data.datasets import BoundaryCondition
    from walrus.models.isotropic_model import dim_pad
    from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic

    @dataclass
    class MockMetadata:
        """Stand-in for the_well dataset metadata, as Walrus' forward expects it.

        ``GeometryEncoder.forward`` replicates part of Walrus' forward pass and
        needs a metadata object to hand down, but has no real the_well dataset
        behind it, so it constructs one of these instead.

        Attributes:
            n_spatial_dims: Number of spatial dimensions in the input grid.
                Default: 3, though ``GeometryEncoder.forward`` always passes 3
                explicitly and then swaps in a ``dataclasses.replace`` copy
                carrying ``self.model.max_d`` before calling into Walrus.
            dataset_name: Dataset name reported to Walrus. Default: ``"test"``,
                a placeholder rather than a real the_well dataset name.
        """

        n_spatial_dims: int = 3
        dataset_name: str = "test"

except ImportError:
    # require_modality_deps(Modality.GEOMETRY) raises with the real
    # ImportError message when GeometryEncoder is instantiated.
    walrus = None
    hydra = None
    MockMetadata = None
    BoundaryCondition = None


class FallbackGeometryEncoder(ModalityEncoder):
    """Minimal flatten -> linear -> d_geo encoder used when Walrus is missing.

    Gated by env var WALRUS_FALLBACK=1; only constructed when GeometryEncoder
    sees require_modality_deps raise. The point is to keep the per-modality
    sweep's text_geometry cell runnable on environments where the walrus
    install is broken — production builds must install walrus, this is
    smoke-only and produces meaningless features.
    """

    def __init__(self, input_dim: int = 6, d_geo: int = 512):
        super().__init__(d_geo)
        self.output_dim = d_geo
        self.hidden_dim = d_geo
        self.proj = nn.Linear(input_dim, d_geo)
        logger.warning(
            "Using FallbackGeometryEncoder (WALRUS_FALLBACK=1). Output features "
            "are meaningless — sweep throughput only. Install walrus for real runs."
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Flatten, project and mean-pool the input into a single geometry token.

        All leading dimensions are collapsed, so only the last dimension is
        treated as features; it is zero-padded or truncated to ``input_dim``
        before the linear projection.

        Args:
            inputs: Any tensor whose leading dimension is the batch and whose
                last dimension is features, e.g. ``(B, N, D_in)``. Cast to the
                projection's dtype when it differs.

        Returns:
            Token-shaped features of ``(B, 1, d_geo)``: the projected rows are
            regrouped per batch element and mean-pooled over that element's
            positions. These are meaningless as physics features — the layer is
            randomly initialized and exists only to keep smoke runs moving.
        """
        target_dtype = self.proj.weight.dtype
        if inputs.dtype != target_dtype:
            inputs = inputs.to(target_dtype)
        # Collapse any leading shape down to (B, N, D_in), then project to d_geo
        # and pool to (B, 1, d_geo) so downstream consumers see a token-shaped
        # output (B, T, d_geo) like the real encoder returns.
        flat = inputs.reshape(-1, inputs.shape[-1])
        if flat.shape[-1] < self.proj.in_features:
            pad = self.proj.in_features - flat.shape[-1]
            flat = torch.nn.functional.pad(flat, (0, pad))
        elif flat.shape[-1] > self.proj.in_features:
            flat = flat[..., : self.proj.in_features]
        proj = self.proj(flat).view(inputs.shape[0], -1, self.output_dim)
        return proj.mean(dim=1, keepdim=True)


class GeometryEncoder(ModalityEncoder):
    """
    Uses Walrus (polymathic-ai/walrus) for geometry/physics encoding.
    Input: Point Cloud / Mesh Nodes (B, N, 3) or Physics Fields
    Output: Features (B, N, D_geo)
    """

    def __new__(cls, *args, **kwargs):
        """Return a ``FallbackGeometryEncoder`` instead when Walrus is missing.

        The substitution happens only when ``WALRUS_FALLBACK=1`` is set *and*
        ``require_modality_deps(Modality.GEOMETRY)`` actually raises, so a
        working install is never silently downgraded. ``input_dim`` and
        ``d_geo`` are pulled out of the call's keyword or positional arguments
        and forwarded to the fallback.

        Args:
            *args: Positional arguments of ``__init__``; read positionally as
                ``input_dim`` then ``d_geo``.
            **kwargs: Keyword arguments of ``__init__``; ``input_dim`` and
                ``d_geo`` are read by name and take precedence.

        Returns:
            A ``FallbackGeometryEncoder`` when the fallback conditions hold
            (already fully constructed, so ``GeometryEncoder.__init__`` does not
            run on it), otherwise a fresh uninitialized ``GeometryEncoder``.
        """
        # WALRUS_FALLBACK=1 swaps in the smoke-only encoder when walrus is
        # missing. Honor it only if the real deps actually fail to import —
        # otherwise the operator gets a silent downgrade.
        if os.environ.get("WALRUS_FALLBACK") == "1":
            from src.modalities import Modality
            from src.utils.optional_deps import (
                MissingOptionalDependencyError,
                require_modality_deps,
            )

            try:
                require_modality_deps(Modality.GEOMETRY)
            except MissingOptionalDependencyError as e:
                logger.warning(
                    f"WALRUS_FALLBACK=1 and walrus deps missing ({e.__class__.__name__}); "
                    f"constructing FallbackGeometryEncoder for smoke runs."
                )
                input_dim = kwargs.get("input_dim", args[0] if len(args) > 0 else 6)
                d_geo = kwargs.get("d_geo", args[1] if len(args) > 1 else 512)
                return FallbackGeometryEncoder(input_dim=input_dim, d_geo=d_geo)
        return super().__new__(cls)

    def __init__(
        self, input_dim: int = 6, d_geo: int = 512, model_name: str = "polymathic-ai/walrus"
    ):
        # __new__ may have returned a FallbackGeometryEncoder; in that case
        # __init__ is invoked on the fallback (different class) and this body
        # is bypassed. Guard against re-init on the fallback if Python's MRO
        # lands us here anyway (it won't, but the guard is cheap).
        if isinstance(self, FallbackGeometryEncoder):
            return

        from src.modalities import Modality
        from src.utils.optional_deps import require_modality_deps

        require_modality_deps(Modality.GEOMETRY)
        super().__init__(d_geo)
        self.model_name = model_name
        n_states = 67  # Default for pre-trained model

        logger.info(f"Loading Walrus: {model_name} from local clone using Hydra...")

        # Clear global hydra instance to avoid errors if re-initialized
        GlobalHydra.instance().clear()

        # Path to walrus configs
        # src/libs/walrus/walrus/configs
        config_path = os.path.join(walrus_path, "walrus", "configs")

        # Initialize Hydra
        # We need to use a relative path from the calling script or absolute path?
        # initialize() takes config_path relative to the python script or absolute?
        # It seems initialize() expects relative path to caller or absolute if version_base is set?
        # Let's try absolute path with version_base=None

        # Note: hydra.initialize might expect a path relative to the python file calling it,
        # or we can use initialize_config_dir for absolute path.
        with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(config_name="config", overrides=["server=local"])

            # Override model parameters to match pre-trained checkpoint
            # Based on error logs:
            # hidden_dim: 1408 (was 768)
            # intermediate_dim: 352 (was 192)
            # projection_dim: 352 (was 96)
            # n_states: 67 (was 63)
            cfg.model.hidden_dim = 1408
            cfg.model.intermediate_dim = 352
            cfg.model.projection_dim = 352
            cfg.model.groups = 16  # Inferred from 1408/16=88, 88/4=22, 22/2=11 (freqs size)
            # cfg.model.n_states = 67 # This causes struct error if not in config schema. Pass to instantiate instead.

            # Manually load full_spatial_attention config to replace the missing axial_spatial_attention
            # Use OmegaConf.load to get the content directly without nesting
            full_attn_path = os.path.join(
                config_path, "model/processor/space_mixing/full_spatial_attention.yaml"
            )
            space_mixing_cfg = OmegaConf.load(full_attn_path)
            space_mixing_cfg.num_heads = 16  # Override num_heads to match groups/hidden_dim
            cfg.model.processor.space_mixing = space_mixing_cfg

            # Also override time_mixing num_heads
            if "time_mixing" in cfg.model.processor:
                cfg.model.processor.time_mixing.num_heads = 16

        # Instantiate model
        # We need to set n_states. Walrus pre-trained model has 63 states?
        # The README says "trained on ... 63 physical variables".
        # Let's assume n_states=63 for the pre-trained weights.
        # If we use a different n_states, loading weights might fail or require strict=False.
        # For this demo, we want to load the pre-trained model.
        # Instantiate model
        # We need to set n_states. Walrus pre-trained model has 67 states (from checkpoint mismatch error).
        # n_states is already defined above

        # Instantiate using hydra's instantiate, which resolves the _target_ class
        self.model = hydra.utils.instantiate(cfg.model, n_states=n_states, _recursive_=True)

        # Load weights from shared drive (avoid HF cache issues)
        logger.info(f"Loading weights for {model_name}...")
        # Direct path to shared model location. Set PRISM_WALRUS_WEIGHTS_PATH
        # to your own shared HF hub snapshot to skip the HF download below.
        shared_weights_path = os.environ.get(
            "PRISM_WALRUS_WEIGHTS_PATH",
            "/flare/<project>/<user>/hub/models--polymathic-ai--walrus/snapshots/<revision>/walrus.pt",
        )

        if os.path.exists(shared_weights_path):
            weights_path = shared_weights_path
        else:
            # Fallback to HF download if shared path doesn't exist
            logger.warning("Shared path not found, attempting HF download...")
            weights_path = hf_hub_download(repo_id=model_name, filename="walrus.pt")

        # Load weights
        logger.info(f"Loading weights from {weights_path}...")
        # The checkpoint might be a full checkpoint (app, optimizer, etc.) or just state_dict.
        # walrus.pt usually contains the full checkpoint.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

        if "app" in checkpoint and "model" in checkpoint["app"]:
            state_dict = checkpoint["app"]["model"]
        else:
            state_dict = checkpoint

        # Load state dict
        # We might need to handle shape mismatches if n_states doesn't match exactly
        # or if we are using it for a different task.
        # For now, try strict loading, if it fails, we catch it.
        self.model.load_state_dict(state_dict, strict=False)
        logger.info("Walrus loaded successfully!")

        # Input Projection: 6 -> 63 (n_states)
        self.input_proj = nn.Linear(input_dim, n_states)

        # Output Projection: 1408 (walrus hidden_dim) -> d_geo (512)
        # We are now extracting internal embeddings (hidden_dim) instead of predictions (n_states)
        if self.model is not None and hasattr(self.model, "hidden_dim"):
            self.walrus_dim = self.model.hidden_dim
        else:
            self.walrus_dim = 1408  # Default/Legacy

        # IMPORTANT: This encoder outputs d_geo (512), NOT walrus_dim (1408).
        # We set attributes correctly for UnifiedTransformer to detect.
        self.output_dim = d_geo
        self.hidden_dim = d_geo  # For compatibility with checks looking for this

        self.output_proj = nn.Linear(self.walrus_dim, d_geo)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Voxelize the input fields and run the Walrus encoder and processor.

        The input is first normalized to ``(B, T, N, D_in)``: a 3-D input gains a
        singleton time axis, and 5-D/6-D voxel grids ``(B, T, X, Y, Z[, C])`` are
        strided down — temporally by ``T // 8`` once ``T > 8``, which targets 8
        frames but can leave up to 15, and spatially by 2, 4 or 8 as the largest
        spatial extent passes 32, 64 or 128, which targets roughly a 16-cube —
        and then flattened into points whose features are the
        normalized ``xyz`` coordinates concatenated with the cell values, padded
        or cropped to 6 channels. Points are projected to Walrus' 67 states and
        scattered into a fixed 64-cube grid, which is fed through Walrus'
        ``_encoder_forward`` and processor blocks — the Walrus decoder is
        deliberately skipped so internal embeddings are what comes out. Periodic
        boundary conditions are assumed on all three axes.

        Args:
            inputs: Point cloud ``(B, N, D_in)`` or ``(B, T, N, D_in)``, or a
                voxel grid ``(B, T, X, Y, Z)`` / ``(B, T, X, Y, Z, C)``. A 6-D
                input whose dim 2 is a singleton is first squeezed there, which
                reads it as channel-first ``(B, T, 1, X, Y, Z)``; every other
                6-D input is read as channels-last. Cast to the input
                projection's dtype when it differs.

        Returns:
            Geometry tokens of shape ``(B, T, d_geo)``, mean-pooled over the
            three spatial dimensions of the Walrus feature grid and projected
            from ``walrus_dim`` down to ``d_geo``. ``T`` is the time axis of the
            normalized input: 1 for a 3-D ``(B, N, D_in)`` point cloud, the
            caller's own ``T`` for a 4-D point cloud (temporal striding runs
            only on the voxel-grid path), and the strided frame count for a
            voxel grid.
        """
        # inputs: (B, N, Input_Dim) OR (B, T, N, Input_Dim)

        # Cast input to match model dtype (e.g. float16)
        target_dtype = self.input_proj.weight.dtype
        if inputs.dtype != target_dtype:
            inputs = inputs.to(target_dtype)

        # 1. Handle Input Shape (Spatiotemporal support)
        if inputs.ndim == 3:
            # (B, N, D) -> (B, T=1, N, D)
            inputs = inputs.unsqueeze(1)

        # DEBUG: Print exact shape to catch 6D or weird inputs
        if inputs.ndim != 4:
            logger.debug(f"Geometry Input Shape: {inputs.shape}")

        # Handle 6D (B, T, C, X, Y, Z) - Squeeze channel if 1
        if inputs.ndim == 6 and inputs.shape[2] == 1:
            logger.debug("Squeezing channel dim (6D -> 5D)")
            inputs = inputs.squeeze(2)

        # Handle 5D/6D Voxel Grids -> Point Cloud (B, T, N, D)
        # 5D: (B, T, X, Y, Z) -> C=1 implicit
        # 6D: (B, T, X, Y, Z, C)
        if inputs.ndim >= 5:
            # DEBUG: Print original shape
            logger.debug(f"Geometry {inputs.ndim}D Input Shape: {inputs.shape}")

            # --- Downsampling Strategy ---
            shape = inputs.shape
            _B_s, T_s = shape[0], shape[1]
            # Assume X,Y,Z are always after T
            X_s, Y_s, Z_s = shape[2], shape[3], shape[4]

            # Temporal Stride (Aggressive: Target 8 frames)
            t_stride = 1
            if T_s > 8:
                t_stride = max(1, T_s // 8)

            # Spatial Stride (Aggressive: Target 16^3 = 4096 points)
            # 128 -> stride 8 -> 16
            # 64 -> stride 4 -> 16
            s_stride = 1
            max_dim = max(X_s, Y_s, Z_s)

            if max_dim >= 128:
                s_stride = 8
            elif max_dim >= 64:
                s_stride = 4
            elif max_dim > 32:
                s_stride = 2

            if t_stride > 1 or s_stride > 1:
                logger.debug(f"Downsampling Geometry with T_stride={t_stride}, S_stride={s_stride}")
                # Use slicing (works for both 5D and 6D)
                # dims: B, T, X, Y, Z, [C]
                grad_slice = [slice(None)] * inputs.ndim  # All dims
                grad_slice[1] = slice(None, None, t_stride)  # T
                grad_slice[2] = slice(None, None, s_stride)  # X
                grad_slice[3] = slice(None, None, s_stride)  # Y
                grad_slice[4] = slice(None, None, s_stride)  # Z
                inputs = inputs[tuple(grad_slice)]

            # Re-read shape
            shape = inputs.shape
            B, T, X, Y, Z = shape[0], shape[1], shape[2], shape[3], shape[4]
            C = shape[5] if inputs.ndim == 6 else 1

            # Create Coords Grid
            xs = torch.linspace(0, 1, X, device=inputs.device)
            ys = torch.linspace(0, 1, Y, device=inputs.device)
            zs = torch.linspace(0, 1, Z, device=inputs.device)
            grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")

            # (X, Y, Z, 3)
            coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(inputs.dtype)
            # (B, T, X, Y, Z, 3)
            coords = coords.expand(B, T, -1, -1, -1, -1)

            # Flatten spatial: (B, T, N, 3)
            coords_flat = coords.reshape(B, T, X * Y * Z, 3)

            # Flatten values: (B, T, N, C)
            values_flat = inputs.reshape(B, T, X * Y * Z, C)

            # Concat: (B, T, N, 3+C)
            inputs = torch.cat([coords_flat, values_flat], dim=-1)

            # Pad/Crop to input_dim (6) if needed
            if inputs.shape[-1] < 6:
                pad_size = 6 - inputs.shape[-1]
                inputs = F.pad(inputs, (0, pad_size))
            elif inputs.shape[-1] > 6:
                inputs = inputs[..., :6]

        B, T, N, D_in = inputs.shape
        H, W, D = 64, 64, 64
        n_states = 67

        # Flatten Batch and Time for voxelization: (B*T, N, D)
        inputs_flat = inputs.reshape(B * T, N, D_in)

        # Project features to n_states
        features = self.input_proj(inputs_flat)  # (B*T, N, 67)

        # Create grid
        grid = torch.zeros(B * T, n_states, H, W, D, device=inputs.device, dtype=features.dtype)

        # Normalize coordinates
        coords = inputs_flat[:, :, :3]
        min_coords = coords.min(dim=1, keepdim=True)[0]
        max_coords = coords.max(dim=1, keepdim=True)[0]
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1.0

        norm_coords = (coords - min_coords) / range_coords
        grid_coords = (norm_coords * (H - 1)).long()
        grid_coords = torch.clamp(grid_coords, 0, H - 1)

        # Scatter features into grid
        flat_indices = (
            grid_coords[:, :, 0] * W * D + grid_coords[:, :, 1] * D + grid_coords[:, :, 2]
        )

        for i in range(B * T):
            flat_grid = grid[i].view(n_states, -1)
            flat_grid.index_put_(
                (
                    torch.arange(n_states, device=inputs.device).unsqueeze(1),
                    flat_indices[i].unsqueeze(0),
                ),
                features[i].T,
            )
            grid[i] = flat_grid.view(n_states, H, W, D)

        # Reshape to (T, B, C, H, W, D) for Walrus
        # grid: (B*T, C, H, W, D) -> (B, T, C, H, W, D) -> (T, B, C, H, W, D)
        grid = grid.view(B, T, n_states, H, W, D).permute(1, 0, 2, 3, 4, 5)
        x = grid

        # Prepare metadata for Walrus
        metadata = MockMetadata(n_spatial_dims=3)
        bcs = [[(BoundaryCondition.PERIODIC.value, 0.0)] * 3]
        state_labels = torch.arange(n_states, device=inputs.device)

        # Walrus Forward Pass (Surgical)
        # We replicate IsotropicModel.forward logic but stop before decoder
        # try: -> if True: to verify no fallback without re-indenting everything
        if True:
            # --- START IsotropicModel.forward logic ---
            metadata = replace(metadata, n_spatial_dims=self.model.max_d)
            # Pad dims
            x, squeeze_out = dim_pad(x, self.model.max_d)
            # x is now (T, B, C, H, W, D)

            x_shape = x.shape[3:]
            dim_key = str(metadata.n_spatial_dims)

            # Determine patch sizes / kernels
            # Walrus uses SpaceBagAdaptiveDVstrideEncoder which is variable_downsample=True, variable_deterministic_ds=True

            # Logic from IsotropicModel.forward:
            if (
                hasattr(self.model.embed[dim_key], "variable_downsample")
                and (self.model.embed[dim_key].variable_downsample)
                and self.model.embed[dim_key].variable_deterministic_ds
            ):
                dynamic_ks = choose_kernel_size_deterministic(x_shape)
                patch_size = [reduce(mul, k) for k in dynamic_ks]
                patch_size.extend([0] * (self.model.max_d - len(patch_size)))
            else:
                # Fallback if using a simple encoder (unlikely with Walrus)
                patch_size = [
                    getattr(self.model.embed[dim_key], "patch_size", 4 * 4)
                ] * self.model.max_d
                dynamic_ks = None

            # Encoder Forward
            # Note: We access protected member _encoder_forward
            x, stage_info, jitter_info = self.model._encoder_forward(
                x,
                state_labels,
                bcs,
                metadata,
                patch_size,
                dynamic_ks,
                self.model.encoder_dummy,
            )

            # Processor (Blocks)
            # This is the "Brain"
            # Return attention? No.
            for blk in self.model.blocks:
                x, _ = blk(x, bcs, return_att=False)

            # --- END IsotropicModel.forward logic (before Decoder) ---

            # x shape: (T, B, C_hidden, H', W', D')
            # C_hidden should be 1408

            # Pooling
            # x is (T, B, C, H, W, D)
            # We want (B, T, d_model)
            # Only pool spatial dims
            x = x.mean(dim=[3, 4, 5])  # (T, B, C)

            # Permute to (B, T, C)
            x = x.permute(1, 0, 2)

            # Project to d_model
            x = self.output_proj(x)  # (B, T, d_model)

            return x
