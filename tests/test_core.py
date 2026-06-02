"""Core-algorithm smoke tests using the analytic DummyBackend (no weights/GPU).

These verify the velocity-first PDG plumbing: schedules, residual clamping,
routing, drift gating, anchor inversion round-trip, and the full edit loop.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_pdg.backends.dummy import DummyBackend
from flow_pdg.config import PipelineConfig
from flow_pdg.routing import route_velocity, to_latent_mask
from flow_pdg.drift import drift_ratio, inject_mode, InjectMode
from flow_pdg.anchor import build_anchor, invert
from flow_pdg.pipeline import EditInputs, TriConditionalFlowPDG


def _img(seed, h=64, w=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g) * 2 - 1


# ---- routing --------------------------------------------------------------
def test_route_velocity_background_preserved():
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = 1.0  # top half foreground
    v_ctrl = torch.ones(1, 4, 8, 8)
    v_anchor = torch.zeros(1, 4, 8, 8)
    out = route_velocity(v_ctrl, v_anchor, mask)
    assert out[..., :4, :].mean() == pytest.approx(1.0)
    assert out[..., 4:, :].mean() == pytest.approx(0.0)


def test_mask_downsample_shape():
    m = torch.ones(1, 1, 64, 64)
    lm = to_latent_mask(m, (8, 8), feather=1.0)
    assert lm.shape == (1, 1, 8, 8)
    assert lm.max() <= 1.0 and lm.min() >= 0.0


# ---- drift ----------------------------------------------------------------
def test_drift_gating_thresholds():
    from flow_pdg.config import InjectionConfig
    cfg = InjectionConfig()
    x = torch.zeros(1, 4, 8, 8)
    near = x + 0.01
    far = x + 100.0
    a = x + 1e-9  # anchor ~0 -> use nonzero anchor
    anchor = torch.ones(1, 4, 8, 8)
    d_small = drift_ratio(anchor + 0.001, anchor)
    d_large = drift_ratio(anchor + 10.0, anchor)
    assert inject_mode(d_small, cfg) == InjectMode.STRONG
    assert inject_mode(d_large, cfg) == InjectMode.OFF


# ---- anchor inversion round-trip -----------------------------------------
def test_inversion_reconstructs():
    be = DummyBackend()
    img = _img(0)
    x0 = be.encode_image(img)
    cond = be.encode_prompt("source")
    sigmas = be.sigmas(20)
    x_T = invert(be, x0, cond, sigmas, method="fireflow")
    # replay forward from x_T under same cond should return near x0
    from flow_pdg.solver import euler_step
    x = x_T
    for i in range(len(sigmas) - 1):
        s, sn = float(sigmas[i]), float(sigmas[i + 1])
        v = be.velocity(x, s, cond).velocity
        x = euler_step(x, v, s, sn)
    err = (x - x0).abs().mean().item()
    assert err < 0.2, f"reconstruction error too high: {err}"


def test_build_anchor_lengths():
    be = DummyBackend()
    be.encode_image(_img(1))
    cond = be.encode_prompt("src")
    from flow_pdg.config import AnchorConfig
    traj = build_anchor(be, be.encode_image(_img(1)), cond, AnchorConfig(num_steps=10), capture_layers=(0, 1))
    assert len(traj.latents) == 11        # N+1 nodes
    assert len(traj.velocities) == 10
    assert traj.kv[0] is not None and 0 in traj.kv[0].layers


# ---- full pipeline (depth-conditioned tri-conditional loop, mask routing) --
def test_pipeline_runs():
    cfg = PipelineConfig(backend="dummy", height=64, width=64, device="cpu")
    cfg.anchor.num_steps = 12
    be = DummyBackend()
    pipe = TriConditionalFlowPDG(be, cfg)
    inp = EditInputs(
        source=_img(2), appearance=_img(3),
        fg_mask=_make_mask(64, 64),
        source_prompt="a giraffe standing on grass",
        target_prompt="a giraffe in a snowy winter field",
        appearance_prompt="zebra stripes",
    )
    out = pipe.run(inp)
    assert out.shape == (1, 3, 64, 64)
    assert torch.isfinite(out).all()


def test_pipeline_injection_disabled_backend():
    # backend without K/V injection support => foreground falls back to v_bg (no crash)
    class NoInject(DummyBackend):
        @property
        def supports_kv_injection(self):
            return False

    cfg = PipelineConfig(backend="dummy", height=64, width=64, device="cpu")
    cfg.anchor.num_steps = 10
    out = TriConditionalFlowPDG(NoInject(), cfg).run(
        EditInputs(_img(4), _img(5), _make_mask(64, 64), "src", "tar", "app")
    )
    assert torch.isfinite(out).all()


def _make_mask(h, w):
    m = torch.zeros(1, 1, h, w)
    m[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
    return m


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
