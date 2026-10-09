import torch
from src.encoders.crystal_graph import CrystalGraphTokenEncoder, PeriodicCrystalMessageLayer


def packed_graph_batch(edge_distance: torch.Tensor | None = None):
    return {
        "atomic_numbers": torch.tensor([6, 8, 14, 14, 8]),
        "edge_index": torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 4, 2]]),
        "edge_distance": (
            edge_distance
            if edge_distance is not None
            else torch.tensor([1.2, 1.2, 2.0, 2.1, 1.8])
        ),
        "graph_batch": torch.tensor([0, 0, 1, 1, 1]),
        "batch_size": 2,
    }


def test_crystal_encoder_emits_fixed_tokens_for_variable_size_graphs():
    encoder = CrystalGraphTokenEncoder(
        hidden_dim=16, num_layers=2, radial_dim=8, num_tokens=3, dropout=0.0
    )

    tokens = encoder(packed_graph_batch())

    assert tokens.shape == (2, 3, 16)
    assert encoder.output_dim == 16
    assert encoder.tokens_per_instance() == 3
    assert torch.isfinite(tokens).all()


def test_crystal_encoder_uses_periodic_edge_distances():
    torch.manual_seed(7)
    encoder = CrystalGraphTokenEncoder(
        hidden_dim=16, num_layers=2, radial_dim=8, num_tokens=2, dropout=0.0
    ).eval()

    short_edges = encoder(packed_graph_batch())
    long_edges = encoder(
        packed_graph_batch(torch.tensor([3.8, 3.8, 4.0, 4.1, 3.9]))
    )

    assert not torch.allclose(short_edges, long_edges)


def test_message_layer_aggregates_over_each_center_own_neighbors():
    """A center atom's output must depend on the neighbors *it* declared.

    ``cif_to_graph`` truncates edges per-center (top-K by distance), which
    only bounds each atom's outgoing edge count. Atom 1 here is not a center
    of any edge (no per-center truncation kept it as a source), but it is
    the declared nearest neighbor of both atom 0 and atom 2. Each of those
    centers must aggregate atom 1's features into its own output; atom 1
    itself must not receive anything back, since it has no outgoing edges.
    """
    layer = PeriodicCrystalMessageLayer(hidden_dim=4, radial_dim=2, dropout=0.0)
    nodes = torch.zeros(3, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 2], [1, 1]])  # centers 0 and 2 each -> neighbor 1
    radial = torch.zeros(2, 2)

    output = layer(nodes, edge_index, radial)
    # LayerNorm output always sums to ~0 regardless of input, so a plain
    # `.sum()` loss has a degenerate (numerically noisy, not truly zero)
    # gradient that isn't a reliable dependency signal. Square first so the
    # loss actually varies with the pre-normalization inputs.
    (output[0] ** 2).sum().backward()

    assert nodes.grad[1].abs().sum() > 0, (
        "center 0's output did not depend on its declared neighbor (atom 1)"
    )
    assert nodes.grad[0].abs().sum() > 0, "center 0's output should also depend on itself"


def test_crystal_encoder_backpropagates_and_accepts_graphs_without_edges():
    encoder = CrystalGraphTokenEncoder(
        hidden_dim=16, num_layers=1, radial_dim=4, num_tokens=2, dropout=0.0
    )
    inputs = {
        "atomic_numbers": torch.tensor([6, 8]),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "edge_distance": torch.empty(0),
        "graph_batch": torch.tensor([0, 1]),
        "batch_size": 2,
    }

    tokens = encoder(inputs)
    tokens.square().mean().backward()

    assert tokens.shape == (2, 2, 16)
    assert encoder.atom_embedding.weight.grad is not None
    assert encoder.queries.grad is not None
