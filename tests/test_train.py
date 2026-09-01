"""What ``autotrainer.train()`` actually does, once a backend is picked.

``test_train_dispatch.py`` pins the routing decision. This file pins the
behavior on the other side of it: that the model really trains, that explicit
overrides survive, that ``epochs`` is honored, and that ``save_path`` writes
something the user can load back.

Those bodies were the least-covered lines in the package (13%, against 82%
overall) despite ``train()`` being the headline API - the CI examples job
smoke-ran them, so "it crashed" was caught, but "it silently did the wrong
thing" was not. Two such wrongs are pinned as ``xfail`` below rather than
asserted as correct.
"""

from __future__ import annotations

import json
import re

import pytest


def _epoch_losses(capsys) -> list[float]:
    """The per-epoch losses train() printed, in order."""
    out = capsys.readouterr().out
    return [float(m) for m in re.findall(r"^epoch \d+/\d+: loss ([\d.]+)", out, re.M)]


@pytest.fixture
def separable():
    """A small, genuinely learnable classification task.

    Deliberately not random noise: a model that fails to learn should fail
    these tests, and on noise every recipe looks identical.
    """
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (256,), generator=g)
    # Class 0 clusters at -1, class 1 at +1, so the task is linearly separable.
    X = torch.randn(256, 4, generator=g) * 0.3 + (y.float().unsqueeze(1) * 2 - 1)
    return DataLoader(TensorDataset(X, y), batch_size=32)


@pytest.fixture
def model():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))


class TestTorchTraining:
    """The PyTorch path: the one that runs its own loop."""

    def test_returns_the_trained_model(self, model, separable):
        torch = pytest.importorskip("torch")
        import autotrainer

        out = autotrainer.train(model, separable, epochs=1, lr=1e-3)
        assert isinstance(out, torch.nn.Module)

    def test_parameters_actually_change(self, model, separable):
        torch = pytest.importorskip("torch")
        import autotrainer

        before = [p.detach().clone() for p in model.parameters()]
        trained = autotrainer.train(model, separable, epochs=1, lr=1e-2)
        after = list(trained.parameters())
        assert any(not torch.equal(b, a.detach().cpu()) for b, a in zip(before, after))

    def test_loss_decreases(self, model, separable, capsys):
        """The whole point. A learnable task plus a sane lr must go down."""
        import autotrainer

        autotrainer.train(model, separable, epochs=4, lr=1e-2)
        losses = _epoch_losses(capsys)
        assert len(losses) == 4
        assert losses[-1] < losses[0], f"loss did not fall: {losses}"

    def test_runs_exactly_the_requested_epochs(self, model, separable, capsys):
        import autotrainer

        autotrainer.train(model, separable, epochs=3, lr=1e-3)
        assert len(_epoch_losses(capsys)) == 3

    def test_explicit_loss_fn_is_used(self, model, separable):
        """Not merely accepted - the object passed must be the one called."""
        torch = pytest.importorskip("torch")
        import autotrainer

        class CountingLoss(torch.nn.CrossEntropyLoss):
            calls = 0

            def forward(self, *a, **kw):
                CountingLoss.calls += 1
                return super().forward(*a, **kw)

        autotrainer.train(model, separable, epochs=1, lr=1e-3, loss_fn=CountingLoss())
        assert CountingLoss.calls == len(separable)

    def test_explicit_lr_overrides_the_range_test(self, model, separable, capsys):
        import autotrainer

        autotrainer.train(model, separable, epochs=1, lr=3e-4)
        out = capsys.readouterr().out
        assert "lr=3.00e-04 (user override)" in out
        assert "LR range test" not in out

    def test_patience_is_accepted_and_ignored(self, model, separable, capsys):
        """Documented as the Keras early-stopping knob. On torch it must be a
        no-op rather than a TypeError, since the signature accepts it."""
        import autotrainer

        autotrainer.train(model, separable, epochs=2, lr=1e-3, patience=1)
        assert len(_epoch_losses(capsys)) == 2

    def test_save_path_writes_a_loadable_state_dict(self, model, separable, tmp_path):
        torch = pytest.importorskip("torch")
        import autotrainer

        path = tmp_path / "model.pt"
        trained = autotrainer.train(model, separable, epochs=1, lr=1e-3, save_path=str(path))
        assert path.exists()
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        # A state_dict, not the module: keys must line up with the model's.
        unwrapped = getattr(trained, "module", trained)
        assert set(loaded) == set(unwrapped.state_dict())

    def test_explicit_optimizer_instance_is_used(self, model, separable, capsys):
        """The passed optimizer must be the one that steps, not a rebuilt default.

        Asserted through its ``state``: torch populates that dict on the first
        step, so a non-empty state proves this instance ran the updates. When
        auto() rebuilt its own AdamW instead, this stayed empty.
        """
        torch = pytest.importorskip("torch")
        import autotrainer

        mine = torch.optim.Adam(model.parameters(), lr=7e-3)
        assert not mine.state
        autotrainer.train(
            model, separable, epochs=1, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=mine
        )
        assert mine.state, "the passed optimizer never stepped"
        out = capsys.readouterr().out
        assert "optimizer=Adam (yours, used as-is)" in out
        # Its lr stands in for the range test, which must not have run.
        assert "lr=7.00e-03 (from your optimizer)" in out

    def test_explicit_lr_applies_to_a_passed_optimizer(self, model, separable):
        """Both given: the lr is applied to their optimizer rather than one of
        the two being silently dropped."""
        torch = pytest.importorskip("torch")
        import autotrainer

        # momentum, so there is per-parameter state to prove it stepped -
        # plain SGD keeps none.
        mine = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
        autotrainer.train(
            model,
            separable,
            epochs=1,
            lr=1e-3,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=mine,
        )
        assert mine.state
        # The scheduler moves lr from there, so check the base, not the current.
        assert all(g["initial_lr"] == 1e-3 for g in mine.param_groups)

    def test_cpu_path_trains_without_a_gradscaler(self, model, separable, monkeypatch, capsys):
        """The no-CUDA branch. It is the only one CPU-only CI ever executes, and
        the only one this GPU box otherwise never would."""
        torch = pytest.importorskip("torch")
        import autotrainer

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        autotrainer.train(model, separable, epochs=2, lr=1e-2)
        losses = _epoch_losses(capsys)
        assert len(losses) == 2 and losses[-1] < losses[0]


class TestSklearn:
    """The estimator path: prepare() for n_jobs, then a single fit()."""

    def test_returns_a_fitted_estimator(self):
        pytest.importorskip("sklearn")
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        import autotrainer

        X = np.random.RandomState(0).randn(64, 4)
        y = (X[:, 0] > 0).astype(int)
        est = autotrainer.train(LogisticRegression(), X, y)
        # fit() ran: the attribute only exists afterwards.
        assert hasattr(est, "coef_")
        assert est.score(X, y) > 0.8

    def test_save_path_roundtrips(self, tmp_path):
        pytest.importorskip("sklearn")
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        import autotrainer

        X = np.random.RandomState(0).randn(64, 4)
        y = (X[:, 0] > 0).astype(int)
        path = tmp_path / "est.joblib"
        est = autotrainer.train(LogisticRegression(), X, y, save_path=str(path))
        # Round-trip of a file this test just wrote into tmp_path - the pickle
        # being loaded is the one train() produced a line earlier, not input.
        assert (joblib.load(path).predict(X) == est.predict(X)).all()

    def test_unsupervised_estimator_fits_without_targets(self):
        """y=None is the unsupervised branch: fit(X) rather than fit(X, y)."""
        pytest.importorskip("sklearn")
        import numpy as np
        from sklearn.cluster import KMeans

        import autotrainer

        X = np.random.RandomState(0).randn(64, 4)
        est = autotrainer.train(KMeans(n_clusters=2, n_init=10), X)
        assert est.cluster_centers_.shape == (2, 4)


class TestBoosting:
    """XGBoost/LightGBM via their sklearn API - routed to the boosting backend
    for thread config, then saved like any other estimator."""

    def test_xgboost_sklearn_api_fits(self):
        pytest.importorskip("xgboost")
        import numpy as np
        from xgboost import XGBClassifier

        import autotrainer

        X = np.random.RandomState(0).randn(64, 4)
        y = (X[:, 0] > 0).astype(int)
        clf = autotrainer.train(XGBClassifier(n_estimators=5), X, y)
        assert clf.get_booster().num_boosted_rounds() == 5

    @pytest.mark.xfail(
        reason="the sklearn-API branch always joblib.dumps, so a .json save_path "
        "(what examples/xgboost_example.py passes, and what the example docstrings "
        "advertise) gets a pickle with a .json name that xgboost cannot load back.",
        strict=True,
    )
    def test_json_save_path_writes_xgboost_json(self, tmp_path):
        pytest.importorskip("xgboost")
        import numpy as np
        from xgboost import XGBClassifier

        import autotrainer

        X = np.random.RandomState(0).randn(64, 4)
        y = (X[:, 0] > 0).astype(int)
        path = tmp_path / "clf.json"
        autotrainer.train(XGBClassifier(n_estimators=5), X, y, save_path=str(path))
        json.loads(path.read_text())


class TestNativeXGBoost:
    """A params dict plus a DMatrix - no estimator, and `epochs` means rounds."""

    def test_epochs_becomes_num_boost_round(self):
        xgb = pytest.importorskip("xgboost")
        import numpy as np

        import autotrainer

        X = np.random.RandomState(0).randn(64, 4)
        y = (X[:, 0] > 0).astype(int)
        dtrain = xgb.DMatrix(X, label=y)
        booster = autotrainer.train({"objective": "binary:logistic"}, dtrain, epochs=7)
        assert isinstance(booster, xgb.Booster)
        assert booster.num_boosted_rounds() == 7

    def test_save_path_writes_real_json(self, tmp_path):
        xgb = pytest.importorskip("xgboost")
        import numpy as np

        import autotrainer

        X = np.random.RandomState(0).randn(64, 4)
        y = (X[:, 0] > 0).astype(int)
        dtrain = xgb.DMatrix(X, label=y)
        path = tmp_path / "booster.json"
        autotrainer.train({"objective": "binary:logistic"}, dtrain, epochs=3, save_path=str(path))
        assert "learner" in json.loads(path.read_text())
        reloaded = xgb.Booster()
        reloaded.load_model(str(path))
        assert reloaded.num_boosted_rounds() == 3


class TestTFKeras:
    """The TensorFlow path - the only one where `patience` does anything.

    Named for the `tf` substring: CI's test-tf job (the only one that installs
    tensorflow) selects with -k "TFConfig or tf", so a class without it would
    be skipped in `test` and deselected here - i.e. run nowhere.
    """

    @staticmethod
    def _model_and_data():
        tf = pytest.importorskip("tensorflow")
        import numpy as np

        X = np.random.RandomState(0).randn(64, 4)
        y = (X[:, 0] > 0).astype(int)
        model = tf.keras.Sequential(
            [tf.keras.layers.Input((4,)), tf.keras.layers.Dense(2, activation="softmax")]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        return model, X, y

    def test_fits_and_returns_the_model(self):
        model, X, y = self._model_and_data()
        import autotrainer

        assert autotrainer.train(model, X, y, epochs=2) is model
        assert model.history is not None

    def test_patience_installs_early_stopping(self, monkeypatch):
        """`patience` is documented as the Keras-only knob; check it reaches
        model.fit as an EarlyStopping callback rather than being dropped.

        Only that it is installed. It cannot currently fire: train() passes no
        validation data, so the callback monitors a val_loss that never exists
        and Keras warns on every epoch.
        """
        tf = pytest.importorskip("tensorflow")
        model, X, y = self._model_and_data()
        import autotrainer

        seen = {}
        real_fit = model.fit
        monkeypatch.setattr(model, "fit", lambda *a, **kw: seen.update(kw) or real_fit(*a, **kw))
        autotrainer.train(model, X, y, epochs=1, patience=2)
        cbs = seen.get("callbacks", [])
        assert any(isinstance(c, tf.keras.callbacks.EarlyStopping) for c in cbs)

    def test_save_path_writes_a_loadable_keras_model(self, tmp_path):
        tf = pytest.importorskip("tensorflow")
        model, X, y = self._model_and_data()
        import autotrainer

        path = tmp_path / "m.keras"
        autotrainer.train(model, X, y, epochs=1, save_path=str(path))
        assert tf.keras.models.load_model(path).count_params() == model.count_params()


class TestErrors:
    def test_params_dict_without_dmatrix(self):
        import autotrainer

        with pytest.raises(TypeError, match="native-XGBoost path"):
            autotrainer.train({"objective": "binary:logistic"})

    def test_unsupported_model_type(self):
        import autotrainer

        with pytest.raises(TypeError, match="train\\(\\) supports"):
            autotrainer.train("not a model", [1, 2, 3])
