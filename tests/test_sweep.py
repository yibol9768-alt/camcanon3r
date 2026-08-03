from pathlib import Path

import pytest

from camcanon3r.sweep import plan_prediction_sweep


def _prepared(tmp_path: Path) -> tuple[Path, Path]:
    prepared = tmp_path / "prepared"
    output = tmp_path / "outputs"
    for scene in ("room", "kitchen"):
        for variant in ("center_crop_090", "center_crop_060"):
            (prepared / scene / variant).mkdir(parents=True)
    return prepared, output


def test_multiscene_sweep_plan_and_resume(tmp_path: Path) -> None:
    prepared, output = _prepared(tmp_path)
    runs = plan_prediction_sweep(
        prepared,
        output,
        variants=["center_crop_090", "center_crop_060"],
        scenes=["room", "kitchen"],
        resume=False,
        overwrite=False,
    )
    assert [(run.scene, run.variant) for run in runs] == [
        ("room", "center_crop_090"),
        ("room", "center_crop_060"),
        ("kitchen", "center_crop_090"),
        ("kitchen", "center_crop_060"),
    ]

    first = runs[0].output
    first.parent.mkdir(parents=True)
    first.touch()
    first.with_suffix(".json").touch()
    resumed = plan_prediction_sweep(
        prepared,
        output,
        variants=["center_crop_090", "center_crop_060"],
        scenes=["room", "kitchen"],
        resume=True,
        overwrite=False,
    )
    assert resumed[0].skip
    assert not any(run.skip for run in resumed[1:])


def test_sweep_rejects_existing_or_partial_outputs(tmp_path: Path) -> None:
    prepared, output = _prepared(tmp_path)
    target = output / "room" / "center_crop_090.npz"
    target.parent.mkdir(parents=True)
    target.touch()
    with pytest.raises(RuntimeError, match="partial prediction"):
        plan_prediction_sweep(
            prepared,
            output,
            variants=["center_crop_090"],
            scenes=["room"],
            resume=True,
            overwrite=False,
        )

    target.with_suffix(".json").touch()
    with pytest.raises(FileExistsError, match="prediction already exists"):
        plan_prediction_sweep(
            prepared,
            output,
            variants=["center_crop_090"],
            scenes=["room"],
            resume=False,
            overwrite=False,
        )

    overwritten = plan_prediction_sweep(
        prepared,
        output,
        variants=["center_crop_090"],
        scenes=["room"],
        resume=False,
        overwrite=True,
    )
    assert not overwritten[0].skip
