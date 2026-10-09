"""Graph modality encoder: a GraphMAE2-style masked autoencoder over a GAT backbone.

The module holds three pieces: ``sce_loss`` (the scaled-cosine-error
reconstruction loss), ``GAT`` (a stack of PyG ``GATConv`` layers), and
``GraphMAE2`` (the masked autoencoder that pairs two ``GAT`` stacks).
``GraphEncoder`` wraps them behind the ``ModalityEncoder`` contract and is what
the backbone consumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except ImportError:
    # require_modality_deps(Modality.GRAPH) raises with a clear install hint
    # when GraphEncoder is actually instantiated; keep the module importable.
    GATConv = None
import logging

from .base import ModalityEncoder

logger = logging.getLogger(__name__)


# --- Loss Function ---
def sce_loss(x, y, alpha=3):
    """Compute the scaled cosine error between two batches of vectors.

    Both inputs are L2-normalized along the last dimension, so the per-row error
    is ``(1 - cosine_similarity(x, y)) ** alpha``.

    Args:
        x: Predicted vectors, ``(N, D)``; normalized along the last dimension.
        y: Target vectors, ``(N, D)``; normalized along the last dimension.
        alpha: Exponent sharpening the penalty on poorly aligned rows. Larger
            values downweight rows that are already close. Default: 3.

    Returns:
        Scalar tensor: the mean of the per-row scaled cosine errors.
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


# --- GAT Backbone ---
class GAT(nn.Module):
    """Stack ``num_layers`` PyG ``GATConv`` layers into a graph encoder or decoder.

    With ``num_layers == 1`` the stack is a single ``GATConv`` from ``in_dim`` to
    ``out_dim`` with ``nhead_out`` heads. Otherwise it is an input projection
    (``in_dim`` -> ``num_hidden``, ``nhead`` heads), ``num_layers - 2`` hidden
    layers, and an output projection to ``out_dim`` with ``nhead_out`` heads.
    ``concat_out`` is passed to every layer as ``GATConv(concat=...)``: when
    True each layer concatenates its heads, so a layer declaring ``C`` channels
    emits ``C * heads``, and the following layer's declared ``in_channels`` is
    widened to ``num_hidden * nhead`` to match. Note that widening always uses
    ``nhead``, so when ``nhead_out != nhead`` the final layer's output width is
    ``out_dim * nhead_out``, which is not what a further layer would expect.

    Args:
        in_dim: Width of the input node features.
        num_hidden: Per-head channel count of the hidden layers. Unused when
            ``num_layers == 1``.
        out_dim: Per-head channel count of the final layer.
        num_layers: Number of ``GATConv`` layers in the stack.
        nhead: Attention heads on the input and hidden layers.
        nhead_out: Attention heads on the final layer.
        activation: Callable applied between layers (not after the last one).
            Falsy values skip the activation entirely.
        feat_drop: Dropout passed to each ``GATConv`` as its ``dropout``.
        attn_drop: Accepted for signature compatibility; this implementation
            does not use it.
        negative_slope: LeakyReLU slope of the attention mechanism, passed to
            each ``GATConv``.
        residual: Accepted for signature compatibility; this implementation
            does not use it.
        norm: Accepted for signature compatibility; this implementation does
            not use it.
        concat_out: Whether each ``GATConv`` concatenates its heads instead of
            averaging them. Default: False.
        encoding: Accepted for signature compatibility; this implementation
            does not use it. Default: False.

    Attributes:
        gat_layers: The ``nn.ModuleList`` of ``GATConv`` layers, in order.
        head: ``nn.Identity`` placeholder; ``forward`` does not apply it.
    """

    def __init__(
        self,
        in_dim,
        num_hidden,
        out_dim,
        num_layers,
        nhead,
        nhead_out,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        norm,
        concat_out=False,
        encoding=False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        hidden_in = in_dim
        hidden_out = out_dim

        if num_layers == 1:
            self.gat_layers.append(
                GATConv(
                    hidden_in,
                    hidden_out,
                    heads=nhead_out,
                    dropout=feat_drop,
                    negative_slope=negative_slope,
                    concat=concat_out,
                )
            )
        else:
            # Input projection
            self.gat_layers.append(
                GATConv(
                    hidden_in,
                    num_hidden,
                    heads=nhead,
                    dropout=feat_drop,
                    negative_slope=negative_slope,
                    concat=concat_out,
                )
            )

            # Hidden layers
            for _l in range(1, num_layers - 1):
                # PyG GATConv handles input dim * heads internally if concat=True?
                # No, PyG expects input dim. If previous layer concat=True, input is num_hidden * nhead.
                in_channels = num_hidden * nhead if concat_out else num_hidden
                self.gat_layers.append(
                    GATConv(
                        in_channels,
                        num_hidden,
                        heads=nhead,
                        dropout=feat_drop,
                        negative_slope=negative_slope,
                        concat=concat_out,
                    )
                )

            # Output projection
            in_channels = num_hidden * nhead if concat_out else num_hidden
            self.gat_layers.append(
                GATConv(
                    in_channels,
                    hidden_out,
                    heads=nhead_out,
                    dropout=feat_drop,
                    negative_slope=negative_slope,
                    concat=concat_out,
                )
            )

        self.head = nn.Identity()

    def forward(self, x, edge_index):
        """Run the node features through every ``GATConv`` layer in order.

        Args:
            x: Node features, ``(N, in_dim)``, for one flattened graph.
            edge_index: COO edge list, ``(2, E)``, indexing into ``x``.

        Returns:
            Node embeddings from the last ``GATConv``: ``(N, out_dim *
            nhead_out)`` when ``concat_out`` is True and ``(N, out_dim)`` when
            it is False. The activation is applied after every layer except the
            last, and ``self.head`` is not applied at all.
        """
        h = x
        for layer_idx, layer in enumerate(self.gat_layers):
            h = layer(h, edge_index)
            if layer_idx < self.num_layers - 1:  # Activation for hidden layers
                if self.activation:
                    h = self.activation(h)
        return h


# --- GraphMAE2 Model ---
class GraphMAE2(nn.Module):
    """Masked graph autoencoder: mask node features, re-encode, reconstruct them.

    ``pretrain_loss`` implements the self-supervised objective — a random subset
    of nodes has its input features replaced by ``enc_mask_token``, the whole
    graph is encoded, the masked rows are re-masked with ``dec_mask_token`` in
    latent space, and a one-layer GAT decoder reconstructs the original features
    at those rows under ``sce_loss``. ``forward`` bypasses all of that and just
    returns encoder embeddings for the unmasked graph, which is what
    ``GraphEncoder`` uses at inference time.

    The encoder is a ``GAT`` with ``concat_out=True`` and
    ``nhead_out == nhead``, whose per-head width is ``num_hidden // nhead``, so
    its output width is ``num_hidden`` (exactly, only when ``nhead`` divides
    ``num_hidden``). The decoder is a one-layer ``GAT`` with ``concat_out=False``
    and ``nhead_out=1``, taking ``num_hidden`` back down to ``in_dim``.

    Args:
        in_dim: Width of the input node features, and of the reconstruction
            target.
        num_hidden: Width of the encoder output / latent space. Should be
            divisible by ``nhead``.
        num_layers: Number of layers in the encoder ``GAT``.
        nhead: Attention heads in the encoder and decoder.
        activation: Activation between GAT layers. Default: ``None``, which
            substitutes a fresh ``nn.PReLU()``.
        feat_drop: Dropout inside each ``GATConv``. Default: 0.1.
        attn_drop: Forwarded to ``GAT``, which does not use it. Default: 0.1.
        negative_slope: LeakyReLU slope of the GAT attention. Default: 0.2.
        residual: Forwarded to ``GAT``, which does not use it. Default: False.
        norm: Forwarded to ``GAT``, which does not use it. Default: ``None``.
        mask_rate: Fraction of nodes masked by ``pretrain_loss``. Default: 0.3.
        replace_rate: Stored as ``_replace_rate`` but unread by this
            implementation, which has no "replace with a random node" noise
            branch. Default: 0.1.

    Attributes:
        encoder: The ``GAT`` producing ``(N, num_hidden)`` node embeddings.
        decoder: The one-layer ``GAT`` reconstructing ``(N, in_dim)`` features.
        enc_mask_token: Learned ``(1, in_dim)`` vector substituted for masked
            input features.
        dec_mask_token: Learned ``(1, num_hidden)`` vector substituted for
            masked latents before decoding.
        encoder_to_decoder: Bias-free ``num_hidden -> num_hidden`` linear bridge
            between encoder and decoder.
    """

    def __init__(
        self,
        in_dim,
        num_hidden,
        num_layers,
        nhead,
        activation=None,
        feat_drop=0.1,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=False,
        norm=None,
        mask_rate=0.3,
        replace_rate=0.1,
    ):
        super().__init__()
        if activation is None:
            activation = nn.PReLU()

        self._mask_rate = mask_rate
        self._replace_rate = replace_rate
        self._output_hidden_size = num_hidden

        # Encoder
        self.encoder = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden // nhead,
            out_dim=num_hidden // nhead,  # Output of encoder usually projected later
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
            encoding=True,
        )

        # Decoder (Simplified for this implementation)
        self.decoder = GAT(
            in_dim=num_hidden,
            num_hidden=num_hidden // nhead,
            out_dim=in_dim,  # Reconstruct input features
            num_layers=1,
            nhead=nhead,
            nhead_out=1,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=False,  # Output dim = in_dim
            encoding=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, num_hidden))
        self.encoder_to_decoder = nn.Linear(num_hidden, num_hidden, bias=False)

        # Initialize tokens
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)

    def encoding_mask_noise(self, x, mask_rate=0.3):
        """Replace the features of a random node subset with ``enc_mask_token``.

        Nodes are shuffled once and the first ``int(mask_rate * num_nodes)`` are
        masked: their rows are zeroed and then set to ``enc_mask_token``. The
        input is not mutated.

        Args:
            x: Node features, ``(N, in_dim)``.
            mask_rate: Fraction of nodes to mask. Default: 0.3.

        Returns:
            Tuple ``(out_x, (mask_nodes, keep_nodes))``: ``out_x`` is the masked
            copy of ``x``, ``(N, in_dim)``; ``mask_nodes`` holds the indices that
            were masked and ``keep_nodes`` the remaining indices, together
            covering every node exactly once.
        """
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        out_x[mask_nodes] = 0.0
        out_x[mask_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def forward(self, x, edge_index):
        """Encode the graph without masking.

        Args:
            x: Node features, ``(N, in_dim)``.
            edge_index: COO edge list, ``(2, E)``.

        Returns:
            Encoder node embeddings, ``(N, num_hidden)``. No masking,
            reconstruction or loss is involved; use ``pretrain_loss`` for the
            masked-autoencoding objective.
        """
        # Default forward returns encoder embeddings
        z = self.encoder(x, edge_index)
        return z

    def pretrain_loss(self, x, edge_index):
        """Compute the masked-feature reconstruction loss for one graph.

        Masks ``self._mask_rate`` of the nodes with ``enc_mask_token``, encodes
        the masked graph, re-masks those same rows in latent space with
        ``dec_mask_token`` after the ``encoder_to_decoder`` bridge, decodes, and
        scores the reconstruction against the *original* features at the masked
        rows only. Unmasked rows do not contribute to the loss.

        Args:
            x: Original (unmasked) node features, ``(N, in_dim)``.
            edge_index: COO edge list, ``(2, E)``. The full edge set is used in
                both passes; only features are masked, never edges.

        Returns:
            Scalar ``sce_loss`` between the decoder output and the original
            features at the masked nodes.
        """
        # Masking
        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, self._mask_rate)

        # Encoding
        enc_rep = self.encoder(use_x, edge_index)

        # Decoding
        rep = self.encoder_to_decoder(enc_rep)
        rep[mask_nodes] = 0  # Re-mask for decoder? GraphMAE2 does re-masking.
        rep[mask_nodes] += self.dec_mask_token

        recon = self.decoder(rep, edge_index)

        # Reconstruction Loss
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = sce_loss(x_rec, x_init)

        return loss


# --- Graph Encoder Wrapper ---
class GraphEncoder(ModalityEncoder):
    """
    Uses GraphMAE2 (GAT backbone) for graph encoding.
    Input: Node Features (B, Num_Nodes, Input_Dim) + Adjacency
    Output: Node Features (B, Num_Nodes, D_graph)
    """

    def __init__(
        self, input_dim: int = 32, d_graph: int = 512, num_heads: int = 4, num_layers: int = 2
    ):
        from src.modalities import Modality
        from src.utils.optional_deps import require_modality_deps

        require_modality_deps(Modality.GRAPH)
        super().__init__(d_graph)
        self.d_graph = d_graph

        # GraphMAE2 Model
        logger.info(f"Loading Graph Encoder: GraphMAE2 (Internal) with d_graph={d_graph}...")
        self.model = GraphMAE2(
            in_dim=input_dim, num_hidden=d_graph, num_layers=num_layers, nhead=num_heads
        )
        logger.info("Successfully loaded Graph Encoder: GraphMAE2")

        # Fallback projection for when no edges are provided (MLP)
        self.fallback_proj = nn.Linear(input_dim, d_graph)

    def forward(self, inputs) -> torch.Tensor:
        """Encode a graph (or a padded batch of graphs) into node token features.

        A dense ``(B, N, D)`` node tensor is flattened to ``(B * N, D)``.
        ``edge_index`` may arrive as a stacked ``(B, 2, E)`` tensor or a list of
        per-graph ``(2, E_i)`` tensors, in which case graph ``i``'s edges are
        offset by ``i * N`` and all of them concatenated, so the batch becomes
        one disconnected graph for PyG; empty per-graph entries are dropped. A
        bare ``(2, E)`` tensor is passed through unchanged and is therefore
        expected to already index into the flattened batch.

        Args:
            inputs: Either a dict read under the keys ``"x"`` (node features,
                ``(B, N, D)`` or ``(N, D)``) and ``"edge_index"``, or a bare
                tensor of node features, which is treated as ``x`` with no
                edges.

        Returns:
            Node features of shape ``(B, N, d_graph)``. Whenever edges are
            present these come from ``GraphMAE2.forward`` (encoder embeddings,
            no masking); when ``edge_index`` is ``None`` or absent they come
            from the edge-free ``fallback_proj`` linear layer instead. A 2-D
            ``(N, d_graph)`` result is unsqueezed to a batch of one.
        """
        # inputs: Dict with 'x' and 'edge_index'

        if isinstance(inputs, dict):
            x = inputs.get("x")
            edge_index = inputs.get("edge_index")
        else:
            # Fallback for tensor input
            x = inputs
            edge_index = None

        if edge_index is not None:
            # GraphMAE2 Forward
            # Note: PyG GATConv expects (N, D) input. If B>1, we need to handle batching.
            # For this demo, we assume x is (B, N, D).
            # We can flatten to (B*N, D) and adjust edge_index if it's batched.
            # Or if B=1, it's fine.
            # If B>1 and edge_index is for single graph, we might need to repeat/offset.
            # For simplicity, we'll assume B=1 or independent processing if possible.
            # But GATConv doesn't support (B, N, D) directly without batched edge_index.

            # Let's assume standard PyG batching: x is (Total_Nodes, D), edge_index is (2, Total_Edges).
            # If input x is (B, N, D), we flatten it.

            if x.dim() == 3:
                B, N, D = x.shape
                x_flat = x.view(-1, D)

                # x_flat is (B*N, D)

                # Check if edge_index got stacked into specific 3D tensor (B, 2, E)
                if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 3:
                    # e.g. (16, 2, 0) or (16, 2, E)
                    # Convert to list to use the logic below
                    edge_index = list(edge_index.unbind(0))

                if isinstance(edge_index, list):
                    # Coalesce list of edge_indices into single batch edge_index
                    # We must offset the node indices for each graph in the batch
                    # because we flattened x into a single big graph.
                    # Graph i's nodes start at i * N (since x is padded to fixed N)
                    batch_edges = []
                    for i, edge_tensor in enumerate(edge_index):
                        if edge_tensor.numel() > 0:
                            # edge_tensor is (2, E_i). Add offset.
                            offset = i * N
                            # Ensure device match
                            edge_tensor = edge_tensor.to(x.device)
                            batch_edges.append(edge_tensor + offset)

                    if batch_edges:
                        edge_index = torch.cat(batch_edges, dim=1)
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

                # Now edge_index is (2, TotalEdges) or compatible tensor

                # Let's try to run GAT.
                if isinstance(edge_index, torch.Tensor):
                    logger.debug(
                        f"GAT edge_index shape={edge_index.shape}, device={edge_index.device}"
                    )
                    logger.debug(f"GAT x_flat shape={x_flat.shape}, device={x_flat.device}")
                out = self.model(x_flat, edge_index)
                out = out.view(B, N, -1)  # Reshape back
            else:
                # x is (N, D)
                out = self.model(x, edge_index)

                # Ensure output is 3D (Batch=1, Nodes, Dim)
                if out.dim() == 2:
                    out = out.unsqueeze(0)

        else:
            # Fallback MLP
            out = self.fallback_proj(x)
            if out.dim() == 2:
                out = out.unsqueeze(0)

        return out
