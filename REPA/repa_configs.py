"""Fixed multi-layer REPA configurations (--repa-config).

All layer numbers are 1-indexed transformer block numbers: student layer j is the
SiT hidden state after ``blocks[j - 1]``; teacher layer i is the teacher's patch tokens
after ``blocks[i - 1]``, post-processed like the existing baseline target, so the final
teacher layer is exactly that target (utils.forward_teacher_layers). The presets A-E are
defined for a 12-block teacher (our SatDINO-B); a single pair S<j>T<i> works with any
supported teacher. Usage examples are in JOURNAL.md.
"""
import math
import re
import zlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

STUDENT_LAYERS = (6, 8, 10, 12, 14, 16)
TEACHER_DEPTH = 12  # the presets assume a 12-block teacher (SatDINO ViT-S/B)
TEACHER_LAYERS = (3, 6, 9, 12)
FINAL_TEACHER_LAYER = TEACHER_DEPTH
MEAN = 'mean'  # Marks S_mean / T_mean: LN-normalized mean over all STUDENT_LAYERS or TEACHER_LAYERS.


@dataclass(frozen=True)
class Connection:
    student: object  # int layer or MEAN
    teacher: object  # int layer or MEAN
    weight: float

    @property
    def name(self):
        """Stable projector key, e.g. 's08_t12', 's08_tmean', 'smean_tmean'."""
        s = 'smean' if self.student == MEAN else f's{self.student:02d}'
        t = 'tmean' if self.teacher == MEAN else f't{self.teacher:02d}'
        return f'{s}_{t}'

    @property
    def target(self):
        return 'tmean' if self.teacher == MEAN else f't{self.teacher:02d}'


@dataclass(frozen=True)
class RepaConfig:
    id: str
    description: str
    connections: tuple
    teacher_depth: int = TEACHER_DEPTH

    @property
    def final_teacher_layer(self):
        return self.teacher_depth

    @property
    def teacher_mean_layers(self):
        """Teacher layers averaged by T_mean (presets only, so a 12-block teacher)."""
        return TEACHER_LAYERS

    @property
    def student_layers(self):
        """Student hidden states the model must expose."""
        layers = set()
        for c in self.connections:
            layers.update(STUDENT_LAYERS if c.student == MEAN else (c.student,))
        return tuple(sorted(layers))

    @property
    def teacher_layers(self):
        """Teacher layers extracted in one teacher forward pass."""
        layers = set()
        for c in self.connections:
            layers.update(self.teacher_mean_layers if c.teacher == MEAN else (c.teacher,))
        return tuple(sorted(layers))

    def summary(self):
        return {
            'repa_config': self.id,
            'teacher_depth': self.teacher_depth,
            'description': self.description,
            'student_layers': list(self.student_layers),
            'teacher_layers': list(self.teacher_layers),
            'connections': {c.name: c.weight for c in self.connections},
            'num_projectors': len(self.connections),
        }


def _uniform(pairs, description, config_id, teacher_depth):
    weight = 1.0 / len(pairs)
    connections = tuple(Connection(s, t, weight) for s, t in pairs)
    return RepaConfig(config_id, description, connections, teacher_depth)


def _build_configs():
    s, t, d = STUDENT_LAYERS, TEACHER_LAYERS, TEACHER_DEPTH
    return {
        'A': _uniform([(j, 12) for j in s], 'Final teacher: every student -> T12', 'A', d),
        'B': _uniform([(j, 9) for j in s], 'Intermediate teacher: every student -> T9', 'B', d),
        # 1/24 per pair == (1/6) sum_j (1/4) sum_i, since every student has 4 teachers.
        'C': _uniform([(j, i) for j in s for i in t], 'All-to-all: 24 pair projectors', 'C', d),
        'D': _uniform([(j, MEAN) for j in s], 'Teacher mean: every student -> T_mean', 'D', d),
        'E': _uniform([(MEAN, MEAN)], 'Global means: S_mean -> T_mean, one projector', 'E', d),
    }


REPA_CONFIGS = _build_configs()


SINGLE_PAIR = re.compile(r'S(\d+)T(\d+)')


def get_repa_config(config_id, teacher_depth=TEACHER_DEPTH):
    """A preset (A-E) or a single pair 'S<j>T<i>' (student block j -> teacher block i),
    for a teacher with ``teacher_depth`` blocks (utils.teacher_depth)."""
    if config_id in REPA_CONFIGS:
        if teacher_depth != TEACHER_DEPTH:
            raise ValueError(f"--repa-config {config_id}: presets A-E are defined for a "
                             f"{TEACHER_DEPTH}-block teacher (SatDINO-B), this teacher has "
                             f"{teacher_depth} blocks; use a single pair S<j>T<i>")
        return REPA_CONFIGS[config_id]
    m = SINGLE_PAIR.fullmatch(config_id or '')
    if m:
        j, i = int(m.group(1)), int(m.group(2))
        if j < 1 or not 1 <= i <= teacher_depth:
            raise ValueError(f"--repa-config {config_id}: student block must be >= 1 and teacher block "
                             f"in 1..{teacher_depth} (the teacher has {teacher_depth} blocks)")
        return _uniform([(j, i)], f'Single pair: S{j} -> T{i}', config_id, teacher_depth)
    raise ValueError(f"Unknown --repa-config {config_id!r}; use a preset ({','.join(REPA_CONFIGS)}) "
                     "or a single pair S<j>T<i>, e.g. S8T9")


def layer_norm(x):
    """Parameter-free LayerNorm over each token's channels, computed in fp32."""
    return F.layer_norm(x.float(), (x.shape[-1],))


def ln_mean(features):
    """Mean of individually LayerNormed [B, N, D] tensors (normalize before averaging)."""
    shapes = {tuple(f.shape) for f in features}
    if len(shapes) != 1:
        raise ValueError(f"Cannot average features of different shapes: {sorted(shapes)}")
    return torch.stack([layer_norm(f) for f in features]).mean(dim=0)


def build_teacher_targets(config, teacher_features):
    """Map each connection target name to a detached [B, N, D] teacher tensor.

    teacher_features: {1-indexed teacher layer: [B, N, D]} from one frozen forward pass.
    """
    missing = set(config.teacher_layers) - set(teacher_features)
    if missing:
        raise ValueError(f"Teacher layers {sorted(missing)} were not extracted")
    targets = {}
    for c in config.connections:
        if c.target in targets:
            continue
        if c.teacher == MEAN:
            targets[c.target] = ln_mean([teacher_features[i] for i in config.teacher_mean_layers]).detach()
        else:
            targets[c.target] = teacher_features[c.teacher].detach()
    return targets


def project_students(config, hidden_states, projectors):
    """Apply each connection's projector to its student state; returns {connection name: [B, N, Dz]}."""
    outputs = {}
    student_mean = None
    for c in config.connections:
        if c.student == MEAN:
            if student_mean is None:
                student_mean = ln_mean([hidden_states[j] for j in STUDENT_LAYERS])
            h = student_mean.to(hidden_states[STUDENT_LAYERS[0]].dtype)
        else:
            h = hidden_states[c.student]
        N, T, D = h.shape
        outputs[c.name] = projectors[c.name](h.reshape(-1, D)).reshape(N, T, -1)
    return outputs


def connection_losses(config, projections, targets):
    """Per-example REPA loss of every connection, with the existing loss convention:
    negative cosine similarity per patch token, averaged over tokens -> [B]."""
    losses = {}
    for c in config.connections:
        z_tilde = projections[c.name]
        z = targets[c.target]
        if z_tilde.shape[:2] != z.shape[:2] or z_tilde.shape[-1] != z.shape[-1]:
            raise ValueError(f"{c.name}: projection {tuple(z_tilde.shape)} vs target {tuple(z.shape)}")
        z_tilde = F.normalize(z_tilde, dim=-1)
        z = F.normalize(z, dim=-1)
        losses[c.name] = -(z * z_tilde).sum(dim=-1).mean(dim=-1)
    return losses


def reduce_losses(config, losses):
    """Weighted mean over connections; the weights of every config sum to 1."""
    return sum(c.weight * losses[c.name] for c in config.connections)


def log_terms(config, losses):
    """Scalar per-connection and per-student-layer means for logging."""
    terms = {f'conn/{name}': value.detach().mean() for name, value in losses.items()}
    by_student = {}
    for c in config.connections:
        if c.student != MEAN:
            by_student.setdefault(c.student, []).append(losses[c.name].detach().mean())
    for j, values in by_student.items():
        if len(by_student) > 1 or len(values) > 1:
            terms[f'student/s{j:02d}'] = torch.stack(values).mean()
    return terms


def projector_seed(name, seed):
    return zlib.crc32(f'repa-projector/{name}/seed={seed}'.encode())


@torch.no_grad()
def init_projector(mlp, name, seed):
    """Xavier-uniform weights and zero biases (the SiT convention), drawn from a generator
    keyed by connection name, so e.g. s08_t12 starts identically in configs 0, A and B."""
    generator = torch.Generator().manual_seed(projector_seed(name, seed))
    for module in mlp.modules():
        if isinstance(module, torch.nn.Linear):
            fan_out, fan_in = module.weight.shape
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            weight = torch.empty(module.weight.shape, dtype=torch.float32)
            weight.uniform_(-bound, bound, generator=generator)
            module.weight.copy_(weight)
            if module.bias is not None:
                module.bias.zero_()
