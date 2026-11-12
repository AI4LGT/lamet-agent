"""Worker recycling must preserve ordered fits and captured chained posteriors."""

import os

import gvar as gv
import numpy as np

import lamet_agent.parallel._pool as pool_module
from lamet_agent.parallel._pool import _ParallelPool
from lamet_agent.parallel.fitting import _prior_payload, _sample_fit


def _linear_model(x, p):
    return p["offset"] + p["slope"] * x


def _fit_and_worker(task):
    return os.getpid(), _sample_fit(task)


def _chained_maps():
    x = np.arange(5.0)
    covariance = 0.01 * np.eye(len(x)) + 0.002
    prior = gv.BufferDict({"offset": gv.gvar(0, 3), "slope": gv.gvar(0, 3)})
    previous = None
    stages, executors = [], []
    with _ParallelPool(2) as pool:
        for stage in range(3):
            # Empty calls must not consume the lifetime of an existing pool.
            assert pool.map(_fit_and_worker, []) == []
            tasks = []
            for sample in range(6):
                sample_prior = prior
                if previous is not None:
                    posterior = previous[sample][1][3]
                    sample_prior = gv.BufferDict(
                        {key: gv.gvar(gv.mean(posterior[key]), 2 * gv.sdev(posterior[key])) for key in prior}
                    )
                mean = 1.0 + sample * 0.01 + (0.7 + stage * 0.01) * x
                tasks.append(
                    (
                        x,
                        mean,
                        covariance,
                        _linear_model,
                        _prior_payload(sample_prior),
                        {"svdcut": 1e-12, "maxit": 500},
                        True,
                    )
                )
            values = pool.map(_fit_and_worker, tasks)
            executors.append(pool._executor)
            records = []
            for pid, (parameters, error, quality, posterior, sdevs) in values:
                assert error is None
                assert gv.evalcov(posterior.buf)[0, 1] != 0
                records.append(
                    {
                        "pid": pid,
                        "parameters": dict(parameters),
                        "quality": quality,
                        "posterior_mean": gv.mean(posterior.buf).tolist(),
                        "posterior_covariance": gv.evalcov(posterior.buf).tolist(),
                        "sdevs": {key: value.tolist() for key, value in sdevs.items()},
                    }
                )
            stages.append(records)
            previous = values
    return stages, executors


def test_recycling_preserves_serialized_posteriors_and_dependent_priors(monkeypatch):
    monkeypatch.setattr(pool_module, "_MAX_MAPS_PER_EXECUTOR", 128)
    persistent, persistent_executors = _chained_maps()
    assert all(executor is persistent_executors[0] for executor in persistent_executors)
    monkeypatch.setattr(pool_module, "_MAX_MAPS_PER_EXECUTOR", 2)
    recycled, recycled_executors = _chained_maps()
    assert recycled_executors[0] is not None
    assert recycled_executors[1] is None
    assert recycled_executors[2] is not None
    assert recycled_executors[0] is not recycled_executors[2]
    before = {record["pid"] for stage in recycled[:2] for record in stage}
    after = {record["pid"] for record in recycled[2]}
    assert before.isdisjoint(after)
    for original_stage, recycled_stage in zip(persistent, recycled):
        for original, restarted in zip(original_stage, recycled_stage):
            original.pop("pid")
            restarted.pop("pid")
            assert original == restarted


def test_recycling_does_not_start_workers_for_serial_maps(monkeypatch):
    def unexpected_executor(**_kwargs):
        raise AssertionError("serial execution must not start a process pool")

    monkeypatch.setattr(pool_module, "_MAX_MAPS_PER_EXECUTOR", 1)
    monkeypatch.setattr(pool_module, "ProcessPoolExecutor", unexpected_executor)
    with _ParallelPool(1) as pool:
        for _ in range(3):
            assert pool.map(abs, [-2, -1]) == [2, 1]
            assert pool._executor is None
            assert pool._completed_maps == 0
