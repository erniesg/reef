"""Per-scenario adapter routing on a shared-base training runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from reef_service.runtime_stubs import StubInferenceRuntime, StubTrainingRuntime

from reef.artifact import Artifact, ArtifactRef, InMemoryRepositoryBackend, LiveWeightArtifactRef
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, ReleaseComponents
from reef.core.errors import ReefError
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import adapter_name, create_weight_surface
from reef.surface.base import CheckpointRecoveryRuntime, ComponentSurface, Surface
from reef.surface.weights import WeightInferenceHooks, WeightLoader, artifact_runtime_load_id


def live(scenario_version: str) -> Artifact:
    return Artifact(
        LiveWeightArtifactRef(
            content_id="live:x", release_id="live:p:1", parent_release_id=None, runtime_load_id=scenario_version
        ),
        None,
    )


def checkpoint(tmp_path: Path, runtime_load_id: str | None) -> Artifact:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    return Artifact.local(root, metadata={} if runtime_load_id is None else {"runtime_load_id": runtime_load_id})


def test_live_requests_route_to_the_scenario_publication() -> None:
    hooks = WeightInferenceHooks(scenario="math")
    out = hooks.prepare_request(live("inc:7"), "/v1/chat/completions", {"messages": []})
    assert out["lora_path"] == adapter_name("math", "inc:7")
    assert out["return_meta_info"] is True
    with pytest.raises(ReefError, match="asked for lora_path"):
        hooks.prepare_request(live("inc:7"), "/v1/chat/completions", {"lora_path": adapter_name("code", "inc:7")})


def test_checkpoint_requests_route_by_recorded_runtime_load_id(tmp_path: Path) -> None:
    hooks = WeightInferenceHooks(scenario="math")
    out = hooks.prepare_request(checkpoint(tmp_path, "inc:3"), "/v1/chat/completions", {})
    assert out["lora_path"] == adapter_name("math", "inc:3")


def test_unpublished_scenarios_sample_the_base(tmp_path: Path) -> None:
    hooks = WeightInferenceHooks(scenario="math")
    out = hooks.prepare_request(checkpoint(tmp_path, None), "/v1/chat/completions", {"messages": []})
    assert "lora_path" not in out
    with pytest.raises(ReefError, match="published no adapter yet"):
        hooks.prepare_request(checkpoint(tmp_path, None), "/v1/chat/completions", {"lora_path": "x"})


def test_shared_and_per_scenario_modes_are_exclusive() -> None:
    with pytest.raises(ValueError, match="either"):
        WeightInferenceHooks("reef_lora", scenario="math")
    assert artifact_runtime_load_id(ArtifactRef("id", "v", None)) is None


def test_recovery_checks_the_scenario_adapter_not_the_global_version() -> None:
    class Runtime(StubTrainingRuntime):
        def serving_runtime_load_id(self):
            return "inc:9"  # another scenario published since

        def serving_adapter_runtime_load_id(self, scenario):
            return {"math": "inc:4"}.get(scenario)

    current = LiveWeightArtifactRef(
        content_id="live:x", release_id="live:p:4", parent_release_id=None, runtime_load_id="inc:4"
    )
    checkpoint_ref = ArtifactRef("ckpt", "c0", None)
    runtime = Runtime().inference
    assert WeightLoader("math").recover(current, checkpoint_ref, runtime) == current
    assert WeightLoader().recover(current, checkpoint_ref, runtime) == checkpoint_ref
    # The engine holds no adapter for code: its live head is unservable, so
    # serving falls back to the exact checkpoint instead of routing to a
    # name the engine would reject.
    assert WeightLoader("code").recover(current, checkpoint_ref, runtime) == checkpoint_ref
    surface = create_weight_surface(scenario="math")
    assert isinstance(surface.loader, WeightLoader) and isinstance(surface.inference, WeightInferenceHooks)


def test_a_restarted_engine_gets_the_recovered_head_loaded_back(tmp_path: Path) -> None:
    """Recovery decides; activation has to act on the decision.

    A runtime that keeps its weights inside the Reef process loses them when
    that process exits. `recover` already spots that — the engine reports a
    runtime load ID from a new incarnation — and falls serving back to the
    checkpoint. Before this, nothing then put the checkpoint into the engine:
    the scenario reported its full step count and kept training from the bare
    base model, with no record anywhere that it had.
    """
    restored: list[str] = []

    class Runtime(StubTrainingRuntime, CheckpointRecoveryRuntime):
        def serving_runtime_load_id(self):
            return "mlx-222-1"  # a fresh process: counter back at one

        def restore_recovered_checkpoint(self, artifact):
            restored.append(str(artifact.local_path))
            return "mlx-222-2"

    current = LiveWeightArtifactRef(
        content_id="live:x", release_id="live:p:40", parent_release_id=None, runtime_load_id="mlx-111-40"
    )
    ckpt = checkpoint(tmp_path, "mlx-111-40")
    loader = WeightLoader()

    assert loader.recover(current, ckpt.ref, Runtime()) == ckpt.ref
    assert loader.restore_recovered(ckpt, Runtime()) == "mlx-222-2"
    assert restored == [str(ckpt.local_path)]

    # A runtime that now serves it needs nothing more.
    class Loaded(Runtime):
        def serving_runtime_load_id(self):
            return "mlx-111-40"

    assert loader.restore_recovered(ckpt, Loaded()) is None
    assert len(restored) == 1


def test_an_artifact_with_no_recorded_version_is_left_alone(tmp_path: Path) -> None:
    # An unknown published version is not a sign of a stale engine, and a
    # runtime that reports none of its own cannot be compared against.
    class Runtime(StubTrainingRuntime, CheckpointRecoveryRuntime):
        def serving_runtime_load_id(self):
            return "mlx-111-40"

        def restore_recovered_checkpoint(self, artifact):  # pragma: no cover - must not run
            raise AssertionError("a publication must not reload weights from disk")

    assert WeightLoader().restore_recovered(checkpoint(tmp_path, None), Runtime()) is None


def test_a_runtime_with_no_known_version_is_not_reloaded(tmp_path: Path) -> None:
    class UnknownRuntime(StubInferenceRuntime, CheckpointRecoveryRuntime):
        def restore_recovered_checkpoint(self, artifact):
            raise AssertionError("an unknown serving version is not a positive mismatch")

    runtime = UnknownRuntime(StubTrainingRuntime(), base_url="http://unknown")
    assert WeightLoader().restore_recovered(checkpoint(tmp_path, "inc:40"), runtime) is None


@pytest.mark.parametrize("returned_version", [None, "", 42])
def test_startup_restore_requires_a_nonempty_runtime_version(tmp_path: Path, returned_version: object) -> None:
    class Runtime(StubInferenceRuntime, CheckpointRecoveryRuntime):
        def serving_runtime_load_id(self):
            return "inc:1"

        def restore_recovered_checkpoint(self, artifact):
            return returned_version

    runtime = Runtime(StubTrainingRuntime(), base_url="http://checkpoint-recovery")
    with pytest.raises(TypeError, match="restore_recovered_checkpoint must return a non-empty runtime load ID"):
        WeightLoader().restore_recovered(checkpoint(tmp_path, "inc:40"), runtime)


def test_surface_restores_only_the_weight_component_with_its_metadata(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (tmp_path / "skills").mkdir()
    restored: list[tuple[Path, str]] = []

    class Runtime(StubInferenceRuntime, CheckpointRecoveryRuntime):
        def serving_adapter_runtime_load_id(self, scenario):
            return "inc:3" if scenario == "math" else "inc:9"

        def restore_recovered_checkpoint(self, artifact):
            restored.append((artifact.local_path, artifact.metadata["runtime_load_id"]))
            return "inc:5"

    artifact = Artifact.local(
        tmp_path,
        metadata={
            "runtime_load_id": "other-component-version",
            COMPONENTS_METADATA_KEY: ReleaseComponents(
                {
                    "weights": ComponentEntry("weights:4", {"runtime_load_id": "inc:4"}),
                    "skills": ComponentEntry("skills:7"),
                }
            ).to_dict(),
        },
    )
    surface = Surface(
        components={"weights": ComponentSurface(loader=WeightLoader("math")), "skills": ComponentSurface()}
    )
    runtime = Runtime(StubTrainingRuntime(), base_url="http://components")
    surface.restore_recovered(artifact, runtime)
    assert restored == [(weights, "inc:4")]
    # Normal publication still uses activation, which must not reload bytes.
    surface.activate(artifact, runtime)
    assert restored == [(weights, "inc:4")]


def test_a_matching_engine_keeps_serving_the_live_head(tmp_path: Path) -> None:
    # Same process, same weights: recovery leaves the live head in place and
    # activation stays out of the way.
    class Runtime(StubTrainingRuntime, CheckpointRecoveryRuntime):
        def serving_runtime_load_id(self):
            return "mlx-111-40"

        def restore_recovered_checkpoint(self, artifact):  # pragma: no cover - must not run
            raise AssertionError("an unchanged engine must not be reloaded")

    current = LiveWeightArtifactRef(
        content_id="live:x", release_id="live:p:40", parent_release_id=None, runtime_load_id="mlx-111-40"
    )
    loader = WeightLoader()
    assert loader.recover(current, ArtifactRef("ckpt", "c0", None), Runtime()) == current
    assert loader.restore_recovered(checkpoint(tmp_path, "mlx-111-40"), Runtime()) is None


def test_a_head_that_is_its_own_checkpoint_still_gets_restored(tmp_path: Path) -> None:
    """`checkpoint_every_n_versions: 1` makes every publication a checkpoint.

    The head's release id then equals the checkpoint's, and recovery used to
    return on that alone — before asking whether the engine still held those
    weights. That is not an edge case: for any deployment checkpointing every
    version it is the only case, and it silently served the bare base model
    after every restart while reporting the full step count.
    """
    restored: list[str] = []

    class Runtime(StubTrainingRuntime, CheckpointRecoveryRuntime):
        def serving_runtime_load_id(self):
            return "mlx-222-1"

        def restore_recovered_checkpoint(self, artifact):
            restored.append(str(artifact.local_path))
            return "mlx-222-2"

    same_release = "live:p:40"
    current = LiveWeightArtifactRef(
        content_id="live:x", release_id=same_release, parent_release_id=None, runtime_load_id="mlx-111-40"
    )
    ckpt = checkpoint(tmp_path, "mlx-111-40")
    loader = WeightLoader()

    # Head and checkpoint are one release, so `recover` short-circuits...
    assert loader.recover(current, ArtifactRef("ckpt", same_release, None), Runtime()).release_id == same_release
    # ...but the question it short-circuits past has already been answered.
    assert loader.restore_recovered(ckpt, Runtime()) == "mlx-222-2"
    assert restored == [str(ckpt.local_path)]


def test_executor_startup_keeps_coordinator_owned_checkpoint_recovery(tmp_path: Path) -> None:
    from reef.inference.model_config import ModelConfig
    from reef.inference.runtime import ExecutorInferenceRuntime
    from reef.observability import NullExperimentTracker
    from reef.scenario.factory import ScenarioFactory
    from reef.service.runtime import connect_executor_runtimes
    from reef.storage.commits import CommitRecord

    from .test_executor_runtime import Coordinator

    class WeightSurfaceRecipe(Recipe):
        def serving_surface(self, scenario):
            return create_weight_surface()

    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    storage = SQLiteScenarioStorage(tmp_path / "state")
    first = ScenarioFactory(
        WeightSurfaceRecipe(), backend_factory, scenario_storage=storage, experiment_tracker=NullExperimentTracker()
    )
    first.load_or_create("math", model_config=ModelConfig()).close()
    backend = backend_factory("math")
    staged = checkpoint(tmp_path, "previous-incarnation:40")
    published = backend.publish(
        Artifact.local(staged.local_path, metadata={**backend.metadata(), **staged.metadata}),
        expected_parent=backend.current(),
        advance_head=False,
    )
    store = storage.open("math")
    store.commit_step(
        expected_step=0,
        commit=CommitRecord(
            scenario="math",
            step=1,
            artifact_ref=published,
            checkpoint=True,
            algorithm_state={"steps": 1},
            high_water_sequence=0,
            high_water_offset=0,
        ),
    )
    store.close()
    control = Coordinator()
    training, runtime = connect_executor_runtimes(train_group_handle=control)
    assert isinstance(runtime, ExecutorInferenceRuntime)
    assert runtime.serving_runtime_load_id() == "engine:0"
    second = ScenarioFactory(
        WeightSurfaceRecipe(runtime=runtime),
        backend_factory,
        scenario_storage=storage,
        experiment_tracker=NullExperimentTracker(),
    )
    scenario = None
    try:
        scenario = second.load_or_create("math", model_config=ModelConfig())
        assert scenario.scenario_step == 1
        assert scenario.repository.require_current_artifact() == published
        assert backend.current() == published
        assert runtime.serving_runtime_load_id() == "engine:0"
        assert runtime.inference_admission_status["open"] is False
        assert control.calls == []
    finally:
        if scenario is not None:
            scenario.close()
        storage.close()
        training.shutdown()
        runtime.shutdown()


@pytest.mark.parametrize("load_fails", [False, True])
@pytest.mark.parametrize("durable_commit", [False, True])
def test_scenario_startup_restores_before_admission_and_fails_closed(
    tmp_path: Path, load_fails: bool, durable_commit: bool
) -> None:
    from reef.inference.model_config import ModelConfig
    from reef.observability import NullExperimentTracker
    from reef.scenario.factory import ScenarioFactory
    from reef.storage.commits import CommitRecord
    from reef.train.mlx_backend.serving import MLXServingRuntime

    class WeightSurfaceRecipe(Recipe):
        def serving_surface(self, scenario):
            return create_weight_surface()

    restored: list[tuple[Path, str]] = []
    acknowledgments: list[tuple[str, str]] = []

    class Engine:
        publication = 0

        def next_runtime_load_id(self):
            self.publication += 1
            return f"mlx-fresh-{self.publication}"

        def load_adapter(self, path):
            assert runtime.inference_admission_status["open"] is False
            assert runtime.current_runtime_load_id() == "mlx-fresh-1"
            if load_fails:
                raise RuntimeError("adapter load failed")

    class FreshRuntime(MLXServingRuntime):
        def release(self):
            assert self.inference_admission_status["open"] is False
            acknowledgments.append((self.serving_runtime_load_id(), self.current_runtime_load_id()))
            super().release()

        def restore_checkpoint(self, artifact):
            restored.append((artifact.local_path, artifact.metadata["runtime_load_id"]))
            return super().restore_checkpoint(artifact)

    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    storage = SQLiteScenarioStorage(tmp_path / "state" if durable_commit else None)
    first = Dispatcher(WeightSurfaceRecipe(), backend_factory, scenario_storage=storage)
    first.get_or_create_scenario("math")
    first.close()
    storage = SQLiteScenarioStorage(tmp_path / "state" if durable_commit else None)

    backend = backend_factory("math")
    staged = checkpoint(tmp_path, "mlx-previous-40")
    checkpoint_artifact = Artifact.local(
        staged.local_path,
        metadata={**backend.metadata(), "runtime_load_id": "mlx-previous-40"},
    )
    published = backend.publish(
        checkpoint_artifact, expected_parent=backend.current(), advance_head=not durable_commit
    )
    if durable_commit:
        # The commit is durable before the backend mirrors its chosen head.
        store = storage.open("math")
        store.commit_step(
            expected_step=0,
            commit=CommitRecord(
                scenario="math",
                step=1,
                artifact_ref=published,
                checkpoint=True,
                algorithm_state={"steps": 1},
                high_water_sequence=0,
                high_water_offset=0,
            ),
        )
        store.close()
        assert backend.current() != published

    runtime = FreshRuntime(Engine())
    second = ScenarioFactory(
        WeightSurfaceRecipe(runtime=runtime),
        backend_factory,
        scenario_storage=storage,
        experiment_tracker=NullExperimentTracker(),
    )
    scenario = None
    try:
        if load_fails:
            with pytest.raises(RuntimeError, match="adapter load failed"):
                second.load_or_create("math", model_config=ModelConfig())
            assert runtime.inference_admission_status["open"] is False
            assert runtime.current_runtime_load_id() == "mlx-fresh-1"
            assert acknowledgments == []
        else:
            scenario = second.load_or_create("math", model_config=ModelConfig())
            assert scenario.repository.require_current_artifact() == published
            assert scenario.scenario_step == (1 if durable_commit else 0)
            assert runtime.inference_admission_status["open"] is True
            assert acknowledgments == [("mlx-fresh-2", "mlx-fresh-2")]
        assert restored == [(backend.materialize(published).local_path, "mlx-previous-40")]
        assert backend.current() == published
    finally:
        if scenario is not None:
            scenario.close()
        storage.close()


def test_a_materialized_artifact_carries_the_version_it_was_published_under(tmp_path: Path) -> None:
    """Materializing checks out the manifest beside the bytes; returning only
    the bytes made every caller that needed the record see an empty mapping.

    Restoration after a restart depends entirely on this: with no recorded
    version there is nothing to compare the engine against, so the check
    passes vacuously and the stale engine keeps serving.
    """
    import json

    from reef.artifact.artifact import ArtifactMaterializationError
    from reef.artifact.git_lfs import materialized_metadata

    destination = tmp_path / "cached"
    destination.mkdir()
    (destination / "reef-artifact.json").write_text(
        json.dumps({"content_id": "c", "metadata": {"runtime_load_id": "mlx-111-40"}})
    )
    assert materialized_metadata(destination) == {"runtime_load_id": "mlx-111-40"}

    # Bootstrap trees have no manifest; corrupt durable metadata fails closed.
    assert materialized_metadata(tmp_path / "missing") == {}
    (destination / "reef-artifact.json").write_text("{not json")
    with pytest.raises(ArtifactMaterializationError, match="invalid artifact manifest"):
        materialized_metadata(destination)
