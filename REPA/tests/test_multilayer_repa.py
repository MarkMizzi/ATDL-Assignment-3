"""Tests for the fixed multi-layer REPA configurations (--repa-config).

Run from REPA/:  python -m pytest tests -q      (or: python tests/test_multilayer_repa.py)
Tests that need the real SatDINO weights or the pretrained SiT checkpoint are skipped
when those are unavailable; set REPA_PRETRAINED=/path/to/0400000.pt to run the latter.
"""
import copy
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import repa_configs as rc  # noqa: E402
from repa_configs import MEAN, STUDENT_LAYERS, TEACHER_LAYERS  # noqa: E402

# Every preset plus the two single pairs used in our runs (S8T12 = the original baseline).
REPA_CONFIGS = {cid: rc.get_repa_config(cid) for cid in ['S8T12', 'S8T9', 'A', 'B', 'C', 'D', 'E']}
from models.sit import SiT, build_mlp  # noqa: E402
from loss import SILoss  # noqa: E402

try:
    import pytest
    skip = pytest.skip
except ImportError:  # plain-python runner below
    pytest = None

    class Skipped(Exception):
        pass

    def skip(reason):
        raise Skipped(reason)

HIDDEN, Z_DIM, PROJ_DIM, TOKENS = 64, 24, 32, 16
EXPECTED_COUNTS = {'S8T12': 1, 'S8T9': 1, 'A': 6, 'B': 6, 'C': 24, 'D': 6, 'E': 1}


def small_sit(repa_config=None, z_dims=(), encoder_depth=8, seed=0, randomize=True):
    """SiT with 16 blocks (enough for student layer 16) and tiny widths."""
    torch.manual_seed(seed)
    model = SiT(
        input_size=8, patch_size=2, in_channels=4, hidden_size=HIDDEN, decoder_hidden_size=HIDDEN,
        depth=16, num_heads=4, num_classes=10, use_cfg=True, z_dims=list(z_dims),
        projector_dim=PROJ_DIM, encoder_depth=encoder_depth, repa_config=repa_config,
        repa_z_dim=Z_DIM, repa_seed=seed, fused_attn=False, qk_norm=False,
    )
    if randomize:
        # adaLN-Zero makes a fresh SiT an identity map; emulate a pretrained backbone instead.
        g = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'projectors' not in name and p.requires_grad:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def batch(seed=0, n=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 8, 8, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    teacher = {i: torch.randn(n, TOKENS, Z_DIM, generator=g) * (1 + i) + i for i in TEACHER_LAYERS}
    return x, y, teacher


def retain_hidden_states(model):
    """Forward hooks that keep every block output (1-indexed) with .grad retained."""
    states = {}

    def hook(index):
        def fn(_module, _inputs, output):
            output.retain_grad()
            states[index] = output
        return fn

    handles = [b.register_forward_hook(hook(i + 1)) for i, b in enumerate(model.blocks)]
    return states, handles


# --------------------------------------------------------------------------- structure

def test_connection_and_loss_counts():
    for cid, expected in EXPECTED_COUNTS.items():
        config = REPA_CONFIGS[cid]
        assert len(config.connections) == expected, cid
        assert len({c.name for c in config.connections}) == expected, cid
        model = small_sit(config, randomize=False)
        assert len(model.repa_projectors) == expected, cid
        assert len(model.projectors) == 0, cid
        x, y, teacher = batch()
        targets = rc.build_teacher_targets(config, teacher)
        _, projections = model(x, torch.rand(3), y)
        assert set(projections) == {c.name for c in config.connections}
        losses = rc.connection_losses(config, projections, targets)
        assert len(losses) == expected, cid
    assert list(rc.REPA_CONFIGS) == ['A', 'B', 'C', 'D', 'E']  # report IDs, in order


def test_exact_layers_and_connections():
    assert STUDENT_LAYERS == (6, 8, 10, 12, 14, 16) and TEACHER_LAYERS == (3, 6, 9, 12)
    pairs = lambda cid: [(c.student, c.teacher) for c in REPA_CONFIGS[cid].connections]
    assert pairs('S8T12') == [(8, 12)]
    assert pairs('S8T9') == [(8, 9)]
    assert pairs('A') == [(j, 12) for j in STUDENT_LAYERS]
    assert pairs('B') == [(j, 9) for j in STUDENT_LAYERS]
    assert pairs('C') == [(j, i) for j in STUDENT_LAYERS for i in TEACHER_LAYERS]
    assert pairs('D') == [(j, MEAN) for j in STUDENT_LAYERS]
    assert pairs('E') == [(MEAN, MEAN)]
    assert REPA_CONFIGS['S8T12'].teacher_layers == (12,)
    assert REPA_CONFIGS['S8T9'].teacher_layers == (9,) and REPA_CONFIGS['S8T9'].student_layers == (8,)
    assert REPA_CONFIGS['A'].teacher_layers == (12,)
    assert REPA_CONFIGS['B'].teacher_layers == (9,)  # T9 extracted for C
    assert REPA_CONFIGS['C'].teacher_layers == TEACHER_LAYERS
    assert REPA_CONFIGS['D'].teacher_layers == TEACHER_LAYERS
    assert REPA_CONFIGS['E'].student_layers == STUDENT_LAYERS
    # B vs C (and 0 vs A): identical student attachment points and weights; only the teacher layer differs.
    a, b = REPA_CONFIGS['A'].connections, REPA_CONFIGS['B'].connections
    assert [(c.student, c.weight) for c in a] == [(c.student, c.weight) for c in b]
    assert {c.teacher for c in a} == {12} and {c.teacher for c in b} == {9}
    assert [c.name for c in b] == [f's{j:02d}_t09' for j in STUDENT_LAYERS]
    assert len({c.name for c in REPA_CONFIGS['C'].connections}) == 24  # 24 independent heads
    assert [c.name for c in REPA_CONFIGS['D'].connections] == [f's{j:02d}_tmean' for j in STUDENT_LAYERS]
    assert REPA_CONFIGS['E'].connections[0].name == 'smean_tmean'
    for bad in ('G', '0', 'F', 'S8', 'S8T13', 'S0T9', '8:9'):
        try:
            rc.get_repa_config(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} accepted')
    assert [(c.student, c.teacher) for c in rc.get_repa_config('S10T6').connections] == [(10, 6)]


def test_student_layer_indexing_is_one_based():
    """Student layer j must be the output of blocks[j - 1] (same as --encoder-depth)."""
    config = REPA_CONFIGS['A']
    model = small_sit(config).eval()
    states, handles = retain_hidden_states(model)
    captured = {}
    original = rc.project_students

    def spy(cfg, hidden, projectors):
        captured.update(hidden)
        return original(cfg, hidden, projectors)

    rc.project_students = spy
    try:
        x, y, _ = batch()
        model(x, torch.rand(3), y)
    finally:
        rc.project_students = original
        for h in handles:
            h.remove()
    for j in STUDENT_LAYERS:
        assert captured[j] is states[j], j  # states[j] is the output of blocks[j - 1]


# --------------------------------------------------------------------------- weighting

def test_loss_weighting_is_a_mean():
    for cid, config in REPA_CONFIGS.items():
        assert abs(sum(c.weight for c in config.connections) - 1.0) < 1e-12, cid
        # Distinguishable synthetic losses: L = 100*j + i (student mean -> 0, teacher mean -> 0).
        losses = {
            c.name: torch.tensor([float(100 * (c.student if c.student != MEAN else 0)
                                        + (c.teacher if c.teacher != MEAN else 0))])
            for c in config.connections
        }
        expected = torch.stack(list(losses.values())).mean()
        assert torch.allclose(rc.reduce_losses(config, losses), expected), cid


def test_uniform_all_to_all_nested_equals_flat():
    config = REPA_CONFIGS['C']
    L = {(j, i): 100.0 * j + i for j in STUDENT_LAYERS for i in TEACHER_LAYERS}
    losses = {f's{j:02d}_t{i:02d}': torch.tensor([v]) for (j, i), v in L.items()}
    nested = sum(sum(L[j, i] for i in TEACHER_LAYERS) / 4 for j in STUDENT_LAYERS) / 6
    flat = sum(L.values()) / 24
    got = rc.reduce_losses(config, losses).item()
    assert abs(nested - flat) < 1e-9 and abs(got - flat) < 1e-6 * flat  # float32 sum
    assert abs(got - sum(L.values())) > 1.0  # not 24 full-strength losses
    # Total strength equals one connection's: identical per-connection losses give that loss.
    same = {name: torch.tensor([0.37]) for name in losses}
    assert abs(rc.reduce_losses(config, same).item() - 0.37) < 1e-6


def test_per_student_log_terms():
    config = REPA_CONFIGS['C']
    losses = {c.name: torch.tensor([float(c.student + c.teacher)]) for c in config.connections}
    terms = rc.log_terms(config, losses)
    assert len([k for k in terms if k.startswith('conn/')]) == 24
    for j in STUDENT_LAYERS:
        assert abs(terms[f'student/s{j:02d}'].item() - (j + 7.5)) < 1e-6  # mean of T = 7.5
    assert len([k for k in rc.log_terms(REPA_CONFIGS['D'], {
        c.name: torch.ones(1) for c in REPA_CONFIGS['D'].connections}) if k.startswith('student/')]) == 6


# --------------------------------------------------------------------------- normalization

def test_ln_mean_normalizes_each_layer_before_averaging():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2, 5, 8, generator=g), torch.randn(2, 5, 8, generator=g)
    ref = rc.ln_mean([a, b])
    # Per-layer constant shifts and positive rescaling are removed before the mean
    # (up to LayerNorm's eps=1e-5 in the variance) ...
    assert torch.allclose(rc.ln_mean([3.0 * a + 7.0, b]), ref, atol=1e-4)
    assert torch.allclose(rc.ln_mean([a, 1000.0 * b - 2.0]), ref, atol=1e-4)
    # ... unlike LN(mean), where the large-magnitude layer would dominate.
    post = rc.layer_norm((a + 1000.0 * b) / 2)
    assert not torch.allclose(post, ref, atol=1e-2)
    assert torch.allclose(ref, (rc.layer_norm(a) + rc.layer_norm(b)) / 2)
    assert ref.shape == a.shape


def test_ln_mean_does_not_mix_batch_or_tokens():
    g = torch.Generator().manual_seed(1)
    feats = [torch.randn(3, 6, 8, generator=g) for _ in range(4)]
    ref = rc.ln_mean(feats)
    feats[2] = feats[2].clone()
    feats[2][1, 4] += torch.randn(8, generator=g)
    changed = (rc.ln_mean(feats) - ref).abs().sum(dim=-1) > 1e-6
    expected = torch.zeros(3, 6, dtype=torch.bool)
    expected[1, 4] = True
    assert torch.equal(changed, expected)


def test_ln_mean_rejects_mismatched_shapes():
    try:
        rc.ln_mean([torch.zeros(2, 4, 8), torch.zeros(2, 5, 8)])
    except ValueError:
        return
    raise AssertionError("shape mismatch not detected")


def test_layer_norm_is_parameter_free_and_no_extra_modules():
    for cid in 'DE':
        model = small_sit(REPA_CONFIGS[cid], randomize=False)
        assert not any(isinstance(m, torch.nn.LayerNorm) for m in model.repa_projectors.modules())
        reference = build_mlp(HIDDEN, PROJ_DIM, Z_DIM)
        for mlp in model.repa_projectors.values():
            assert [p.shape for p in mlp.parameters()] == [p.shape for p in reference.parameters()]


def test_teacher_targets_detached_with_valid_shapes():
    g = torch.Generator().manual_seed(0)
    layers = {i: torch.randn(2, TOKENS, Z_DIM, generator=g).requires_grad_() for i in TEACHER_LAYERS}
    for cid, config in REPA_CONFIGS.items():
        targets = rc.build_teacher_targets(config, layers)
        for name, t in targets.items():
            assert not t.requires_grad and t.grad_fn is None, (cid, name)
            assert t.shape == (2, TOKENS, Z_DIM)
    tmean = rc.build_teacher_targets(REPA_CONFIGS['D'], layers)['tmean']
    expected = sum(F.layer_norm(layers[i].detach(), (Z_DIM,)) for i in TEACHER_LAYERS) / 4
    assert torch.allclose(tmean, expected, atol=1e-6)
    assert torch.equal(rc.build_teacher_targets(REPA_CONFIGS['A'], layers)['t12'], layers[12].detach())


# --------------------------------------------------------------------------- initialization

def test_projector_init_is_keyed_by_connection():
    models = {cid: small_sit(REPA_CONFIGS[cid], randomize=False) for cid in ['S8T12', 'A', 'C']}
    ref = models['S8T12'].repa_projectors['s08_t12'].state_dict()
    for cid in 'AC':
        other = models[cid].repa_projectors['s08_t12'].state_dict()
        assert all(torch.equal(ref[k], other[k]) for k in ref), cid
    a = models['C'].repa_projectors['s08_t12'][0].weight
    b = models['C'].repa_projectors['s08_t09'][0].weight
    assert not torch.equal(a, b)
    # Different seeds give different heads.
    other_seed = small_sit(REPA_CONFIGS['S8T12'], seed=1, randomize=False)
    assert not torch.equal(other_seed.repa_projectors['s08_t12'][0].weight,
                           models['S8T12'].repa_projectors['s08_t12'][0].weight)
    bias = models['C'].repa_projectors['s08_t12'][2].bias
    assert torch.count_nonzero(bias) == 0


def test_heads_do_not_change_global_rng_or_backbone_init():
    states = {}
    for cid in [None, 'S8T12', 'C', 'E']:
        model = small_sit(REPA_CONFIGS[cid] if cid else None, randomize=False)
        after = torch.rand(4)  # next global draw, e.g. the first training noise
        backbone = {k: v for k, v in model.state_dict().items() if not k.startswith('repa_projectors.')}
        states[cid] = (backbone, after)
    base_backbone, base_after = states[None]
    for cid in ['S8T12', 'C', 'E']:
        backbone, after = states[cid]
        assert torch.equal(after, base_after), cid
        assert backbone.keys() == base_backbone.keys()
        assert all(torch.equal(backbone[k], base_backbone[k]) for k in backbone), cid


def test_same_random_draws_as_single_layer_run():
    """With the legacy z_dims (as train.py passes them), every config makes exactly the random
    draws of a single-layer REPA run with the same seed: identical backbone and class
    embedding, identical next draw (training noise, data order), and the S8->T12 head starts
    from the single-layer projector's weights."""
    legacy = small_sit(None, z_dims=[Z_DIM], randomize=False)
    legacy_next = torch.rand(4)
    legacy_state = legacy.state_dict()
    legacy_head = {k[len('projectors.0.'):]: v for k, v in legacy_state.items() if k.startswith('projectors.0.')}
    for cid, config in REPA_CONFIGS.items():
        model = small_sit(config, z_dims=[Z_DIM], randomize=False)
        model_next = torch.rand(4)
        assert torch.equal(model_next, legacy_next), cid
        assert len(model.projectors) == 0, cid  # legacy head dropped after construction
        state = model.state_dict()
        for k, v in legacy_state.items():
            if not k.startswith('projectors.'):
                assert torch.equal(state[k], v), (cid, k)
        assert len(model.repa_projectors) == EXPECTED_COUNTS[cid], cid
        if 's08_t12' in model.repa_projectors:
            head = model.repa_projectors['s08_t12'].state_dict()
            assert all(torch.equal(head[k], v) for k, v in legacy_head.items()), cid
    # Other heads keep their connection-keyed init.
    b = small_sit(REPA_CONFIGS['C'], z_dims=[Z_DIM], randomize=False)
    keyed = small_sit(REPA_CONFIGS['C'], randomize=False)
    assert torch.equal(b.repa_projectors['s06_t03'][0].weight, keyed.repa_projectors['s06_t03'][0].weight)


# --------------------------------------------------------------------------- gradients

def repa_backward(config, model):
    x, y, teacher = batch()
    targets = rc.build_teacher_targets(config, teacher)
    states, handles = retain_hidden_states(model)
    _, projections = model(x, torch.rand(3), y)
    for h in handles:
        h.remove()
    loss = rc.reduce_losses(config, rc.connection_losses(config, projections, targets)).mean()
    loss.backward()
    return states


def test_every_projector_gets_gradient():
    for cid, config in REPA_CONFIGS.items():
        model = small_sit(config)
        repa_backward(config, model)
        for name, mlp in model.repa_projectors.items():
            grads = [p.grad for p in mlp.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads), (cid, name)
            assert any(g.abs().sum() > 0 for g in grads), (cid, name)


def test_direct_attachment_points():
    """Gradient w.r.t. leaf student states fed straight into the projectors: isolates direct
    REPA attachment from upstream propagation through later blocks."""
    g = torch.Generator().manual_seed(0)
    for cid, config in REPA_CONFIGS.items():
        model = small_sit(config)
        hidden = {j: torch.randn(3, TOKENS, HIDDEN, generator=g).requires_grad_() for j in range(1, 17)}
        _, _, teacher = batch()
        targets = rc.build_teacher_targets(config, teacher)
        projections = rc.project_students(config, hidden, model.repa_projectors)
        rc.reduce_losses(config, rc.connection_losses(config, projections, targets)).mean().backward()
        attached = {j for j, h in hidden.items() if h.grad is not None and h.grad.abs().sum() > 0}
        assert attached == set(config.student_layers), (cid, attached)
    assert REPA_CONFIGS['S8T12'].student_layers == (8,)
    assert REPA_CONFIGS['E'].student_layers == STUDENT_LAYERS


def test_gradients_through_the_backbone():
    # 0: only S8 is supervised, so blocks 9..16 get exactly zero REPA gradient while block 8
    # and earlier blocks do (distinguishes direct attachment from upstream propagation).
    config = REPA_CONFIGS['S8T12']
    model = small_sit(config)
    states = repa_backward(config, model)
    for j in (10, 12, 14, 16):
        assert states[j].grad is None or states[j].grad.abs().sum() == 0, j
    for index in range(8, 16):  # blocks[8:] == 1-indexed blocks 9..16
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.blocks[index].parameters())
    for j in (6, 8):
        assert states[j].grad.abs().sum() > 0, j
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.blocks[7].parameters())

    # A: block 16 is attached, so it gets REPA gradient; blocks above 16 do not exist here.
    config = REPA_CONFIGS['A']
    model = small_sit(config)
    states = repa_backward(config, model)
    assert any(p.grad.abs().sum() > 0 for p in model.blocks[15].parameters())

    # E: the single aggregate loss reaches all six contributing student states.
    config = REPA_CONFIGS['E']
    model = small_sit(config)
    states = repa_backward(config, model)
    for j in STUDENT_LAYERS:
        assert states[j].grad.abs().sum() > 0, j


def test_teacher_gets_no_gradient():
    teacher_net = torch.nn.Linear(Z_DIM, Z_DIM)
    teacher_net.requires_grad_(False)
    for cid, config in REPA_CONFIGS.items():
        model = small_sit(config)
        x, y, raw = batch()
        with torch.no_grad():
            features = {i: teacher_net(raw[i]) for i in TEACHER_LAYERS}
        targets = rc.build_teacher_targets(config, features)
        _, projections = model(x, torch.rand(3), y)
        rc.reduce_losses(config, rc.connection_losses(config, projections, targets)).mean().backward()
        assert all(p.grad is None for p in teacher_net.parameters()), cid


def test_flow_loss_still_trains_generator():
    config = REPA_CONFIGS['C']
    model = small_sit(config)
    loss_fn = SILoss(repa_config=config)
    x, y, teacher = batch()
    torch.manual_seed(0)
    denoising, proj = loss_fn(model, x, dict(y=y), zs=rc.build_teacher_targets(config, teacher))
    assert torch.isfinite(denoising).all() and torch.isfinite(proj).all()
    assert len(loss_fn.connection_losses) == 24
    denoising.mean().backward()
    for name in ('final_layer.linear.weight', 'blocks.15.mlp.fc2.weight',
                 'x_embedder.proj.weight', 'y_embedder.embedding_table.weight'):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name
    assert all(p.grad is None for p in model.repa_projectors.parameters())


# --------------------------------------------------------------------------- regression

def legacy_and_config0_models():
    legacy = small_sit(None, z_dims=[Z_DIM], encoder_depth=8)
    new = small_sit(REPA_CONFIGS['S8T12'])
    state = legacy.state_dict()
    mapped = {k.replace('projectors.0.', 'repa_projectors.s08_t12.'): v for k, v in state.items()}
    new.load_state_dict(mapped, strict=True)
    return legacy, new


def test_config0_matches_legacy_single_layer_repa():
    legacy, new = legacy_and_config0_models()
    x, y, teacher = batch()
    results = []
    for model, kwargs, zs in [
        (legacy, {}, [teacher[12]]),
        (new, dict(repa_config=REPA_CONFIGS['S8T12']), rc.build_teacher_targets(REPA_CONFIGS['S8T12'], teacher)),
    ]:
        torch.manual_seed(0)
        denoising, proj = SILoss(**kwargs)(model, x, dict(y=y), zs=zs)
        (denoising.mean() + 0.5 * proj.mean()).backward()
        grads = {k.replace('repa_projectors.s08_t12.', 'projectors.0.'): p.grad.clone()
                 for k, p in model.named_parameters() if p.grad is not None}
        results.append((denoising, proj, grads))
    (d0, p0, g0), (d1, p1, g1) = results
    assert torch.equal(d0, d1) and torch.equal(p0, p1)  # bit-identical losses
    assert g0.keys() == g1.keys()
    # Projector gradients are bit-identical; backbone gradients agree to float32 rounding:
    # h_8 feeds both block 9 and the projector, and the legacy forward projects inside the
    # block loop, so autograd sums the two paths in a different order.
    for k in g0:
        if k.startswith('projectors.'):
            assert torch.equal(g0[k], g1[k]), k
        else:
            assert torch.allclose(g0[k], g1[k], rtol=1e-5, atol=1e-9), k


def test_generate_compatible_state_dict():
    """generate.py drops repa_projectors.* and builds no legacy projectors for these checkpoints."""
    model = small_sit(REPA_CONFIGS['B'], randomize=False)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('repa_projectors.')}
    sampler_model = small_sit(None, z_dims=[Z_DIM], randomize=False)
    sampler_model.projectors = torch.nn.ModuleList()
    sampler_model.load_state_dict(state, strict=True)


# --------------------------------------------------------------------------- CLI / loading

def journal_args(*extra, enc='satdino-vit-b'):
    return ['--exp-name=x', '--model=SiT-L/2', f'--enc-type={enc}', '--num-classes=10',
            '--pretrained-path=ckpt.pt', '--learning-rate=1e-5', '--checkpointing-steps=730', *extra]


def test_cli_backwards_compatible():
    import train
    for enc, extra in [('dinov2-vit-l', ()), ('satdino-vit-b', ('--reset-projectors',)),
                       ('dinov2-vit-l,satdino-vit-b', ()), ('none', ())]:
        args = train.parse_args(journal_args(*extra, enc=enc))
        assert args.repa_config is None and not hasattr(args, 'repa_summary'), enc
        assert args.encoder_depth == 8
    args = train.parse_args(journal_args('--reset-projectors', '--repa-config=C'))
    assert args.repa_summary['num_projectors'] == 24
    assert train.parse_args(journal_args('--reset-projectors', '--repa-config=S8T9')).encoder_depth == 8
    for bad in [journal_args('--repa-config=B'),  # missing --reset-projectors
                journal_args('--reset-projectors', '--repa-config=B', enc='dinov2-vit-l'),
                journal_args('--reset-projectors', '--repa-config=A', '--encoder-depth=10'),
                journal_args('--reset-projectors', '--repa-config=S8T9', '--encoder-depth=8'),  # explicit depth
                journal_args('--reset-projectors', '--repa-config=S8T13'),
                journal_args('--reset-projectors', '--repa-config=G')]:  # unknown ID
        try:
            train.parse_args(bad)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")


def test_cli_other_teachers():
    """Single pairs work with any supported teacher; presets A-E only with 12-block teachers."""
    import train
    ok = {('satdino-vit-s', 'C'): (3, 6, 9, 12), ('dinov2-vit-b', 'A'): (12,),
          ('dinov2-vit-l', 'S8T24'): (24,), ('dinov2-vit-l', 'S8T13'): (13,),
          ('dinov2reg-vit-g', 'S8T40'): (40,), ('mocov3-vit-l', 'S6T18'): (18,),
          ('clip-vit-L', 'S8T20'): (20,), ('mae-vit-l', 'S8T24'): (24,), ('jepa-vit-h', 'S8T32'): (32,)}
    for (enc, cid), teacher_layers in ok.items():
        args = train.parse_args(journal_args('--reset-projectors', f'--repa-config={cid}', enc=enc))
        assert tuple(args.repa_summary['teacher_layers']) == teacher_layers, (enc, cid)
    for enc, cid in [('dinov2-vit-l', 'A'), ('dinov2-vit-l', 'E'), ('dinov2-vit-l', 'S8T25'),
                     ('satdino-vit-s', 'S8T13'), ('dinov1-vit-b', 'S8T9'), ('mocov3-resnet-50', 'S8T9'),
                     ('dinov2-vit-l,satdino-vit-b', 'S8T9'), ('none', 'S8T9')]:
        try:
            train.parse_args(journal_args('--reset-projectors', f'--repa-config={cid}', enc=enc))
        except SystemExit:
            continue
        raise AssertionError(f"accepted {enc} {cid}")


def _random_teachers():
    """Randomly initialized teachers of every supported family except SatDINO and DINOv2
    (tested with their real weights), built like load_encoders builds them."""
    from functools import partial
    from models import mocov3_vit, mae_vit, jepa, clip_vit

    class TinyClip(torch.nn.Module):  # the attributes UpdatedVisionTransformer reads from clip's ViT
        def __init__(self, width=64, layers=12, grid=16):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, width, 16, 16, bias=False)
            self.class_embedding = torch.nn.Parameter(torch.randn(width))
            self.positional_embedding = torch.nn.Parameter(torch.randn(grid * grid + 1, width))
            self.ln_pre = torch.nn.LayerNorm(width)
            self.transformer = clip_vit.Transformer(width, layers, 4)

    torch.manual_seed(0)
    mocov3 = mocov3_vit.vit_small()
    mocov3.head = torch.nn.Identity()
    mae = mae_vit.VisionTransformer(num_classes=0, img_size=256, patch_size=16, embed_dim=64, depth=12,
                                    num_heads=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
    ijepa = jepa.vit_tiny(img_size=[256, 256])
    ijepa.forward_features = ijepa.forward
    clip = clip_vit.UpdatedVisionTransformer(TinyClip())
    clip.forward_features = clip.forward
    return {'mocov3': mocov3, 'mae': mae, 'jepa': ijepa, 'clip': clip}


def test_generic_teacher_layers_match_baseline_target():
    from utils import (baseline_teacher_target, check_teacher_layers, forward_teacher_layers,
                       num_teacher_blocks)
    images = torch.randn(2, 3, 256, 256, generator=torch.Generator().manual_seed(0))
    for family, encoder in _random_teachers().items():
        encoder.eval()
        with torch.no_grad():
            assert num_teacher_blocks(encoder, family) == 12, family
            layers = forward_teacher_layers(encoder, family, images, TEACHER_LAYERS)
            baseline = baseline_teacher_target(encoder, family, images)
            check_teacher_layers(encoder, family, images)
        assert torch.equal(layers[12], baseline), family  # final layer == single-layer target
        for i in TEACHER_LAYERS:
            assert layers[i].shape == baseline.shape, (family, i)
        assert not torch.allclose(layers[9], layers[12]), family
        # Hooks are removed: a later plain forward is unaffected and nothing is captured.
        from utils import _teacher_blocks
        assert all(not b._forward_hooks for b in _teacher_blocks(encoder, family)[0]), family


def test_dinov2_layers_match_baseline_target():
    from utils import check_teacher_layers, forward_teacher_layers
    try:
        cached = os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main')
        if os.path.isdir(cached):  # no network round-trip when the hub code is cached
            encoder = torch.hub.load(cached, 'dinov2_vits14_reg', source='local', pretrained=False)
        else:
            encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)
    except Exception as error:  # offline
        skip(f"DINOv2 hub code unavailable: {error}")
    encoder.eval()
    images = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        layers = forward_teacher_layers(encoder, 'dinov2reg', images, TEACHER_LAYERS)
        # Independent reference: DINOv2's own routine (0-indexed, drops class and register tokens).
        reference = encoder.get_intermediate_layers(images, n=[i - 1 for i in TEACHER_LAYERS], norm=True)
        check_teacher_layers(encoder, 'dinov2reg', images)
    for i, ref in zip(TEACHER_LAYERS, reference):
        assert torch.allclose(layers[i], ref, atol=1e-5), i
    assert layers[12].shape == (2, 256, 384)


def test_satdino_small_layers_match_baseline_target():
    from utils import SATDINO_MODEL_IDS, SatDINOEncoder, check_teacher_layers, forward_teacher_layers
    try:
        encoder = SatDINOEncoder(SATDINO_MODEL_IDS['s'])
    except Exception as error:  # offline / not cached
        skip(f"SatDINO-S unavailable: {error}")
    images = torch.randn(2, 3, 256, 256, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        layers = forward_teacher_layers(encoder, 'satdino', images, TEACHER_LAYERS)
        check_teacher_layers(encoder, 'satdino', images)
    assert all(layers[i].shape == (2, 256, 384) for i in TEACHER_LAYERS)
    for cid, config in REPA_CONFIGS.items():
        targets = rc.build_teacher_targets(config, layers)
        assert all(t.shape == (2, 256, 384) for t in targets.values()), cid


def fake_pretrained(tmp_path):
    """A DINOv2-style REPA checkpoint for the small model: legacy projectors.0 of another width."""
    legacy = small_sit(None, z_dims=[Z_DIM + 8], seed=7)
    path = os.path.join(tmp_path, 'pretrained.pt')
    torch.save({'ema': legacy.state_dict(), 'model': legacy.state_dict(),
                'args': SimpleNamespace(enc_type='dinov2-vit-l')}, path)
    return legacy.state_dict(), path


class _Log:
    def info(self, *_):
        pass

    warning = info


def test_reset_projectors_keeps_backbone_and_initializes_heads(tmp_path=None):
    import tempfile
    import train
    tmp_path = tmp_path or tempfile.mkdtemp()
    pretrained, path = fake_pretrained(str(tmp_path))
    for cid in REPA_CONFIGS:
        model = small_sit(REPA_CONFIGS[cid], randomize=False)
        fresh_heads = copy.deepcopy(model.repa_projectors.state_dict())
        fresh_labels = model.y_embedder.embedding_table.weight.detach().clone()
        args = SimpleNamespace(pretrained_path=path, reset_projectors=True, enc_type='satdino-vit-b')
        train.load_pretrained_weights(model, args, _Log())
        state = model.state_dict()
        for k, v in pretrained.items():
            if k.startswith('projectors.') or k == 'y_embedder.embedding_table.weight':
                continue
            assert torch.equal(state[k], v), (cid, k)  # backbone retained exactly
        assert not any(k.startswith('projectors.') for k in state), cid  # DINOv2 head never loaded
        assert all(torch.equal(model.repa_projectors.state_dict()[k], v) for k, v in fresh_heads.items())
        assert torch.equal(model.y_embedder.embedding_table.weight, fresh_labels)

    # Backbone mismatches are never silently ignored.
    broken = {k: v for k, v in pretrained.items() if k != 'blocks.3.attn.qkv.weight'}
    torch.save({'ema': broken}, path)
    model = small_sit(REPA_CONFIGS['B'], randomize=False)
    try:
        train.load_pretrained_weights(model, SimpleNamespace(
            pretrained_path=path, reset_projectors=True, enc_type='satdino-vit-b'), _Log())
    except RuntimeError:
        return
    raise AssertionError("missing backbone key accepted")


def test_checkpoint_resume_and_ema_roundtrip():
    config = REPA_CONFIGS['B']
    model = small_sit(config)
    ema = copy.deepcopy(model)
    import train
    with torch.no_grad():
        for p in model.repa_projectors.parameters():
            p.add_(1.0)
    train.update_ema(ema, model, decay=0.5)
    for name, p in ema.repa_projectors.named_parameters():
        assert torch.allclose(p, model.repa_projectors.get_parameter(name) - 0.5), name
    restored = small_sit(config, randomize=False)
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(restored.state_dict().values(), model.state_dict().values()))


def test_lora_keeps_repa_heads_trainable():
    try:
        import peft  # noqa: F401
    except ImportError:
        skip("peft not installed")
    import train
    model = train.apply_lora(small_sit(REPA_CONFIGS['A'], randomize=False), 4, ['qkv', 'fc1'])
    heads = [p for n, p in model.named_parameters() if '.repa_projectors.' in n]
    assert len(heads) == 6 * 6 and all(p.requires_grad for p in heads)


# --------------------------------------------------------------------------- real models

def _satdino():
    try:
        from utils import SATDINO_MODEL_IDS, SatDINOEncoder
        return SatDINOEncoder(SATDINO_MODEL_IDS['b'])
    except Exception as error:  # offline / not cached
        skip(f"SatDINO unavailable: {error}")


def test_satdino_layers_match_baseline_target():
    encoder = _satdino()
    assert len(encoder.model.blocks) == 12 and encoder.embed_dim == 768
    assert all(not p.requires_grad for p in encoder.parameters())
    g = torch.Generator().manual_seed(0)
    images = torch.randn(2, 3, 256, 256, generator=g)
    with torch.no_grad():
        layers = encoder.forward_layers(images, TEACHER_LAYERS)
        baseline = encoder.forward_features(images)
        # Independent reference: SatDINO's own intermediate-layer routine (keeps CLS, drops GSD).
        reference = encoder.model.forward_intermediate_layers(images, return_all=True)
    assert set(layers) == set(TEACHER_LAYERS)
    assert torch.equal(layers[12], baseline)  # T12 is exactly the existing baseline target
    for i in TEACHER_LAYERS:
        assert layers[i].shape == (2, 256, 768), i
        assert torch.allclose(layers[i], reference[i - 1][:, 1:], atol=1e-5), i  # 1-indexed
    assert not torch.allclose(layers[9], reference[9][:, 1:], atol=1e-3)  # off-by-one detected
    for cid, config in REPA_CONFIGS.items():
        targets = rc.build_teacher_targets(config, layers)
        assert all(t.shape == (2, 256, 768) for t in targets.values()), cid


def test_real_pretrained_checkpoint():
    path = os.environ.get('REPA_PRETRAINED')
    if not path or not os.path.exists(path):
        skip("set REPA_PRETRAINED to the SiT-L/2 400K checkpoint")
    import train
    from models.sit import SiT_models
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    reference = checkpoint['ema']
    del checkpoint
    for cid in ['S8T12', 'B']:
        model = SiT_models['SiT-L/2'](
            input_size=32, num_classes=10, use_cfg=True, z_dims=[768], encoder_depth=8,
            repa_config=REPA_CONFIGS[cid], repa_z_dim=768, repa_seed=0, fused_attn=True, qk_norm=False,
        )
        train.load_pretrained_weights(model, SimpleNamespace(
            pretrained_path=path, reset_projectors=True, enc_type='satdino-vit-b'), _Log())
        state = model.state_dict()
        for k, v in reference.items():
            if not k.startswith('projectors.') and k != 'y_embedder.embedding_table.weight':
                assert torch.equal(state[k], v), (cid, k)
        assert state['y_embedder.embedding_table.weight'].shape[0] == 11  # 10 classes + null
        assert len(model.repa_projectors) == EXPECTED_COUNTS[cid]
        del model, state


if __name__ == '__main__':
    import inspect
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as error:
                if type(error).__name__ in ('Skipped', 'Skipped_'):
                    print(f"SKIP {name}: {error}")
                    continue
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
