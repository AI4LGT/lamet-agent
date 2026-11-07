"""Nonzero fluctuating CS shares its jackknife source with WF and soft matching."""

import json

import gvar as gv
import numpy as np

from lamet_agent.data import EnsembleData, align_sample_lineage
from lamet_agent.kernels.tmd.cg_nlo import hard_factor
from lamet_agent.stages.correlator_analysis._input import resample_correlators
from lamet_agent.stages.matching.tmd import match
from lamet_agent.stages.soft_analysis.physics import extract_cs, extract_intrinsic


def _group(name, configurations):
    raw = EnsembleData(None, "raw", list(np.asarray(configurations)), [], {})
    return resample_correlators(
        {"correlators": {"value": raw}, "configuration_ids": list(map(str, range(raw.n_sample)))},
        mode="jackknife", group=name, bin_size=1, n_boot=None, seed=1, tracked=True,
    )["value"]


def test_correlated_nonzero_cs_cancels_at_matched_reference_momentum(tmp_path):
    moving = _group("wf", [1.7, 1.85, 1.93, 2., 2.05, 2.15, 2.3, 2.4])
    rest = _group("rest", [.7, .85, 1., 1.1, 1.35])
    form_factor = _group("ff", [.08, .09, .095, .105, .11, .12])
    moving, rest = align_sample_lineage(moving, rest)
    x = np.linspace(.05, .95, 181)
    gamma = -.3 + .4 * (moving.values - 2)
    amplitude = moving.values / rest.values
    quasi = []
    for index, momentum in ((8, 3.44), (9, 3.87), (10, 4.30)):
        hard = hard_factor(x, observable="tmdwf", momentum_gev=momentum, mu=2., alpha_s=.293)
        values = (amplitude[:, None] * (x * (1 - x)) ** .8 * hard
                  * np.exp(np.log(momentum / 2.58) * gamma[:, None]))
        attrs = {
            **moving.attrs, "observable": "tmdwf", "parton": "quark", "gfix": "CG",
            "distribution": "quasi", "renormalization_scheme": "ratio", "bT_fm": .24,
            "momentum_gev": momentum, "momentum_index": index,
        }
        quasi.append(EnsembleData(None, "jackknife", list(values), ["x"], {"x": x.tolist()}, attrs))
    cs, _ = extract_cs(quasi, {})
    np.testing.assert_allclose(cs.values[:, 0], gamma, atol=1e-14)
    form_factor.array.attrs.update(observable="form_factor", parton="quark", gfix="CG",
                                   bT_fm=.24, momentum_gev=2.58)
    soft, _ = extract_intrinsic(quasi, [form_factor], cs, {})
    # Serialization and the third independent block must retain the fluctuating
    # CS/WF association when these branches join again during matching.
    soft.to_netcdf(tmp_path / "soft.nc")
    soft = EnsembleData.from_netcdf(tmp_path / "soft.nc")
    matched, _ = match(quasi[0], cs, soft, {
        "observable": "tmdwf", "p_ref_gev": 2.58, "lc_x_ls": {"start": .2, "stop": .8},
    })
    layout = json.loads(matched.attrs["jackknife_layout"])
    assert layout == {"ff": 6, "rest": 5, "wf": 8}
    aligned_ff, matched = align_sample_lineage(form_factor, matched)
    normalized = matched.values / np.sqrt(aligned_ff.values[:, None])
    np.testing.assert_allclose(normalized, np.broadcast_to(normalized[-1], normalized.shape), atol=1e-10)
    # The analytically canceled nuisance groups contribute no variance. An
    # independent oracle retains just the six FF leave-one-out variations.
    ff_samples = np.sqrt(form_factor.values[:, None]) * normalized[-1]
    delta = ff_samples - ff_samples.mean(axis=0)
    expected_covariance = 5 / 6 * delta.T @ delta
    np.testing.assert_allclose(gv.evalcov(matched.gvar), expected_covariance, rtol=1e-9, atol=1e-14)
