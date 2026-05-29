import torch
import torch.nn as nn
import pytest
import math
from transformers.models.qwen3_moe.modular_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from reap.observer import MoETransformerObserver
from reap.observer import Qwen3MoEObserverHookConfig
from reap.metrics import angular_distance

def test_ttm_similarity_matrix_zeros_case():
    # setup
    batch, seq, dim = 1, 2, 3
    num_experts, top_k = 2, 2
    # use the real config signature, not positional args
    config = Qwen3MoeConfig(
        hidden_size=dim,
        intermediate_size=dim,
        moe_intermediate_size=dim,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=False,
    )
    block = Qwen3MoeSparseMoeBlock(config)
    model = nn.Sequential(block)
    observer = MoETransformerObserver(model, hook_config=Qwen3MoEObserverHookConfig())

    # run
    inp = torch.zeros(batch, seq, dim)
    _ = model(inp)

    # verify
    state = observer.report_state()
    ttm = state[0]["ttm_similarity_matrix"]
    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(ttm, expected, atol=1e-6)

def test_ttm_similarity_with_manual_weights():
    import math
    # configure a 2-expert block: hidden_dim=2, intermediate=1, top_k=2, identity activation
    config = Qwen3MoeConfig(
        hidden_size=2,
        intermediate_size=1,
        moe_intermediate_size=1,
        num_experts=2,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        hidden_act="linear",
    )
    block = Qwen3MoeSparseMoeBlock(config)
    # force both experts to be selected
    block.gate.weight.data.zero_()
    # override expert weights so expert0→[4,0], expert1→[0,4] on input [1,1]
    e0, e1 = block.experts
    for e in (e0, e1):
        e.gate_proj.weight.data.fill_(1.0)
        e.up_proj.weight.data.fill_(1.0)
    e0.down_proj.weight.data.copy_(torch.tensor([[1.0],[0.0]]))
    e1.down_proj.weight.data.copy_(torch.tensor([[0.0],[1.0]]))

    model = nn.Sequential(block)
    observer = MoETransformerObserver(model, hook_config=Qwen3MoEObserverHookConfig())

    # run with a single token of [1,1]
    inp = torch.ones(1, 1, 2)
    _ = model(inp)

    state = observer.report_state()
    ttm = state[0]["ttm_similarity_matrix"]
    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(ttm, expected, atol=1e-6)


def test_ttm_similarity_with_90_deg_weights():
    import math
    # configure a 2-expert block: hidden_dim=2, intermediate=1, top_k=2, identity activation
    config = Qwen3MoeConfig(
        hidden_size=2,
        intermediate_size=1,
        moe_intermediate_size=1,
        num_experts=2,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        hidden_act="linear",
    )
    block = Qwen3MoeSparseMoeBlock(config)
    # force both experts to be selected
    block.gate.weight.data.zero_()
    # override expert weights so expert0→[4,0], expert1→[0,4] on input [1,1]
    e0, e1 = block.experts
    for e in (e0, e1):
        e.gate_proj.weight.data.fill_(1.0)
        e.up_proj.weight.data.fill_(1.0)
    e0.down_proj.weight.data.copy_(torch.tensor([[4.0],[0.0]]))
    e1.down_proj.weight.data.copy_(torch.tensor([[0.0],[4.0]]))

    model = nn.Sequential(block)
    observer = MoETransformerObserver(model, hook_config=Qwen3MoEObserverHookConfig())

    # run with a single token of [1,1]
    inp = torch.ones(1, 1, 2)
    _ = model(inp)

    state = observer.report_state()
    ttm = state[0]["ttm_similarity_matrix"]
    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(ttm, expected, atol=1e-6)

@pytest.mark.parametrize(
    "logits, expected_pf",
    [
        # 3 tokens, experts=[e0,e1,e0] ⇒ freq=[2,1], pairwise=[[4,3],[3,2]]
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            torch.tensor([[4, 3], [3, 2]], dtype=torch.long),
        ),
        # 3 tokens, experts=[e1,e0,e1] ⇒ freq=[1,2], pairwise=[[2,3],[3,4]]
        (
            torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2, 3], [3, 4]], dtype=torch.long),
        ),
    ],
)
def test_pairwise_expert_frequency(logits, expected_pf):
    batch, total_tokens = 1, logits.shape[0]
    num_experts, top_k = logits.shape[1], 1
    dim = total_tokens  # we'll pick dim=total_tokens for simplicity

    # build config & block
    config = Qwen3MoeConfig(
        hidden_size=dim,
        intermediate_size=dim,
        moe_intermediate_size=dim,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=False,
    )
    block = Qwen3MoeSparseMoeBlock(config)

    # hook gate to always emit our custom logits
    def _override_logits(module, inp, out):
        return logits
    block.gate.register_forward_hook(_override_logits)

    model = nn.Sequential(block)
    observer = MoETransformerObserver(model, hook_config=Qwen3MoEObserverHookConfig())

    # run through model
    inp = torch.randn(batch, total_tokens, dim)
    _ = model(inp)

    # verify pairwise_expert_frequency in observer state
    state = observer.report_state()
    pf = state[0]["pairwise_expert_frequency"]
    assert torch.equal(pf, expected_pf)

@pytest.mark.parametrize("dim", [1, 5, 10])
def test_angular_distance_all_ones_vs_neg_ones(dim):
    v1 = torch.ones(dim)
    v2 = -torch.ones(dim)
    dist = angular_distance(v1.unsqueeze(0), v2.unsqueeze(1))[0]
    expected = torch.tensor(1.0)
    assert torch.allclose(dist, expected, atol=1e-3)

#--- Test Data Setup for 4-Expert TTM Test ---

# Base distance matrix for 4 experts outputting vectors at 0, 90, 180, 270 degrees.
base_dist_matrix = torch.tensor([
    [0.0, 1.0, 2.0, 1.0],
    [1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, 0.0, 1.0],
    [1.0, 2.0, 1.0, 0.0],
])

# Scenario 1: All experts are used. TTM should be the full distance matrix.
logits1 = torch.tensor([[1.,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], dtype=torch.float32)
expected_ttm1 = base_dist_matrix.clone()

# Scenario 2: One expert is unused. TTM should still be the full distance matrix
# because every pair of experts (i,j) has at least one active expert.
logits2 = torch.tensor([[1.,0,0,0], [1.,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=torch.float32)
expected_ttm2 = base_dist_matrix.clone()

# Scenario 3: Two experts (1 and 3) are unused. freq=[3,0,2,0].
# The TTM entry for a pair (i,j) is 0 if neither expert i nor j is ever chosen.
# This happens for pairs (1,1), (1,3), (3,1), and (3,3).
logits3 = torch.tensor([[1.,0,0,0], [0,0,1,0], [1.,0,0,0], [0,0,1,0], [1.,0,0,0]], dtype=torch.float32)
expected_ttm3 = base_dist_matrix.clone()
expected_ttm3[1, 1] = 0.0 # This is already 0, but explicit for clarity
expected_ttm3[1, 3] = 0.0
expected_ttm3[3, 1] = 0.0
expected_ttm3[3, 3] = 0.0 # This is already 0, but explicit for clarity

# Scenario 4: A more complex routing where all experts are still used.
logits4 = torch.tensor([[1.,0,0,0], [0,1.,0,0], [0,1.,0,0], [0,0,1.,0], [0,0,0,1.], [0,0,0,1.], [0,0,0,1.]])
expected_ttm4 = base_dist_matrix.clone()


@pytest.mark.parametrize(
    "logits, expected_ttm",
    [
        (logits1, expected_ttm1),
        (logits2, expected_ttm2),
        (logits3, expected_ttm3),
        (logits4, expected_ttm4),
    ],
    ids=["all-experts-once", "one-expert-unused", "two-experts-unused", "complex-routing"]
)
def test_ttm_with_four_experts_and_controlled_routing(logits, expected_ttm):
    """
    Tests the TTM calculation with 4 experts and controlled routing.

    This test works by:
    1. Configuring a 4-expert MoE block.
    2. Manually setting expert weights to produce orthogonal/opposite 2D vectors
       ([1,0], [0,1], [-1,0], [0,-1]), ensuring predictable angular distances.
    3. Hooking the router gate to inject specific logits, thereby controlling which
       expert is chosen for each token.
    4. Parameterizing the test with different logit patterns to create various
       expert usage frequencies.
    5. Asserting that the calculated TTM in the observer's state matches the
       pre-calculated expected TTM for that routing scenario.
    """
    # 1. Configure a 4-expert block with a 2D hidden state
    num_experts, top_k, hidden_dim, intermediate_dim = 4, 1, 2, 1
    config = Qwen3MoeConfig(
        hidden_size=hidden_dim,
        intermediate_size=intermediate_dim,
        moe_intermediate_size=intermediate_dim,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=False,
        hidden_act="linear",  # Use linear activation for predictable outputs
    )
    block = Qwen3MoeSparseMoeBlock(config)

    # 2. Control expert outputs to be vectors at 0, 90, 180, 270 degrees
    expert_outputs = [
        torch.tensor([[1.0], [0.0]]),  # 0 deg
        torch.tensor([[0.0], [1.0]]),  # 90 deg
        torch.tensor([[-1.0], [0.0]]), # 180 deg
        torch.tensor([[0.0], [-1.0]]), # 270 deg
    ]

    for i, expert in enumerate(block.experts):
        # These weights ensure a predictable intermediate activation
        expert.gate_proj.weight.data.fill_(1.0)
        expert.up_proj.weight.data.fill_(1.0)
        # Set the down_proj weights to produce the desired output vectors
        expert.down_proj.weight.data.copy_(expert_outputs[i])

    # 3. Control routing by hooking the router gate to return our logits
    def _override_gate_logits(module, inp, out):
        return logits.to(inp[0].device)
    block.gate.register_forward_hook(_override_gate_logits)

    # 4. Setup observer and run the model
    model = nn.Sequential(block)
    observer = MoETransformerObserver(model, hook_config=Qwen3MoEObserverHookConfig())

    batch_size, num_tokens = 1, logits.shape[0]
    inp = torch.ones(batch_size, num_tokens, hidden_dim, dtype=torch.float32)
    _ = model(inp)

    # 5. Verify the TTM from the observer's state
    state = observer.report_state()
    ttm = state[0]["ttm_similarity_matrix"]

    assert torch.allclose(ttm, expected_ttm, atol=1e-6)