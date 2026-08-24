from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from domain import GraphSpec


@dataclass(frozen=True)
class NeuralSCMConfig:
    hidden_layers: tuple[int, int] = (32, 32)
    ensemble_size: int = 10
    max_epochs: int = 250
    patience: int = 20
    learning_rate: float = 1e-3
    validation_share: float = 0.20
    base_seed: int = 20260814

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NeuralSCMResult:
    graph_id: str
    outcome: str
    backend: str
    ate: float
    interval: tuple[float, float]
    ensemble_estimates: tuple[float, ...]
    histories: tuple[dict[str, Any], ...]
    config: NeuralSCMConfig
    note: str = "Neural SCM оценивается только после отдельной идентификации."

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = self.config.as_dict()
        return payload


def _parents(graph: GraphSpec, node: str, outcome: str) -> list[str]:
    result = []
    for source, target in graph.edges:
        if target == node:
            result.append(outcome if source == "Y" else source)
    return list(dict.fromkeys(result))


def _mechanism_order(graph: GraphSpec) -> tuple[str, ...]:
    endogenous = {"L", "D", "Y"}
    incoming: dict[str, set[str]] = {node: set() for node in endogenous}
    for source, target in graph.edges:
        if source in endogenous and target in endogenous:
            incoming[target].add(source)
    order: list[str] = []
    remaining = set(endogenous)
    while remaining:
        ready = sorted(node for node in remaining if not (incoming[node] & remaining))
        if not ready:
            raise ValueError(f"Граф {graph.graph_id} не задаёт ациклический порядок механизмов")
        order.extend(ready)
        remaining.difference_update(ready)
    return tuple(order)


class _SklearnMechanism:
    def __init__(self, hidden: tuple[int, int], seed: int, patience: int, max_epochs: int):
        from sklearn.neural_network import MLPRegressor

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=patience,
            max_iter=max_epochs,
            random_state=seed,
        )
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_scale = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        self.x_mean = np.nanmean(x, axis=0)
        self.x_scale = np.nanstd(x, axis=0)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        x_clean = np.where(np.isfinite(x), x, self.x_mean)
        x_scaled = (x_clean - self.x_mean) / self.x_scale
        self.y_mean = float(np.nanmean(y))
        self.y_scale = max(float(np.nanstd(y)), 1e-8)
        self.model.fit(x_scaled, (y - self.y_mean) / self.y_scale)
        return {
            "epochs": int(self.model.n_iter_),
            "train_loss": [float(value) for value in self.model.loss_curve_],
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.x_mean is not None and self.x_scale is not None
        x_clean = np.where(np.isfinite(x), x, self.x_mean)
        scaled = (x_clean - self.x_mean) / self.x_scale
        return np.asarray(
            self.y_mean + self.y_scale * self.model.predict(scaled), dtype=float
        )


def _fit_sklearn(
    data: pd.DataFrame,
    graph: GraphSpec,
    outcome: str,
    config: NeuralSCMConfig,
) -> NeuralSCMResult:
    order = _mechanism_order(graph)
    estimates: list[float] = []
    histories: list[dict[str, Any]] = []
    for member in range(config.ensemble_size):
        models: dict[str, tuple[_SklearnMechanism, list[str]]] = {}
        member_history: dict[str, Any] = {"member": member, "mechanisms": {}}
        for position, node in enumerate(order):
            target = outcome if node == "Y" else node
            parents = _parents(graph, node, outcome)
            valid = data[target].notna()
            mechanism = _SklearnMechanism(
                config.hidden_layers,
                config.base_seed + 100 * member + position,
                config.patience,
                config.max_epochs,
            )
            history = mechanism.fit(
                data.loc[valid, parents].to_numpy(dtype=float),
                data.loc[valid, target].to_numpy(dtype=float),
            )
            models[node] = (mechanism, parents)
            member_history["mechanisms"][node] = {"parents": parents, **history}

        potential: dict[int, np.ndarray] = {}
        for treatment in (0, 1):
            simulated = data.copy()
            simulated["T"] = treatment
            for node in order:
                target = outcome if node == "Y" else node
                mechanism, parents = models[node]
                simulated[target] = mechanism.predict(simulated[parents].to_numpy(dtype=float))
            potential[treatment] = simulated[outcome].to_numpy(dtype=float)
        estimates.append(float(np.mean(potential[1] - potential[0])))
        histories.append(member_history)

    lower, upper = (
        np.quantile(estimates, [0.025, 0.975])
        if len(estimates) > 1
        else (estimates[0], estimates[0])
    )
    return NeuralSCMResult(
        graph_id=graph.graph_id,
        outcome=outcome,
        backend="scikit-learn MLP fallback",
        ate=float(np.mean(estimates)),
        interval=(float(lower), float(upper)),
        ensemble_estimates=tuple(estimates),
        histories=tuple(histories),
        config=config,
    )


def _fit_torch(
    data: pd.DataFrame,
    graph: GraphSpec,
    outcome: str,
    config: NeuralSCMConfig,
) -> NeuralSCMResult:
    import torch
    from torch import nn

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    order = _mechanism_order(graph)
    estimates: list[float] = []
    histories: list[dict[str, Any]] = []

    class Network(nn.Module):
        def __init__(self, inputs: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(inputs, config.hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(config.hidden_layers[0], config.hidden_layers[1]),
                nn.ReLU(),
                nn.Linear(config.hidden_layers[1], 1),
            )

        def forward(self, values):
            return self.layers(values).squeeze(-1)

    for member in range(config.ensemble_size):
        models: dict[str, tuple[nn.Module, list[str], np.ndarray, np.ndarray, float, float]] = {}
        member_history: dict[str, Any] = {"member": member, "mechanisms": {}}
        for position, node in enumerate(order):
            target = outcome if node == "Y" else node
            parents = _parents(graph, node, outcome)
            valid = data[target].notna()
            x = data.loc[valid, parents].to_numpy(dtype=np.float32)
            y = data.loc[valid, target].to_numpy(dtype=np.float32)
            x_mean = np.nanmean(x, axis=0)
            x_scale = np.nanstd(x, axis=0)
            x_scale[x_scale < 1e-8] = 1.0
            x = np.where(np.isfinite(x), x, x_mean)
            y_mean, y_scale = float(np.mean(y)), max(float(np.std(y)), 1e-8)
            x = (x - x_mean) / x_scale
            y = (y - y_mean) / y_scale

            seed = config.base_seed + 100 * member + position
            torch.manual_seed(seed)
            permutation = np.random.default_rng(seed).permutation(len(x))
            split = max(1, int(len(x) * (1.0 - config.validation_share)))
            train_indices, valid_indices = permutation[:split], permutation[split:]
            if len(valid_indices) == 0:
                valid_indices = train_indices[-1:]
            x_tensor = torch.tensor(x)
            y_tensor = torch.tensor(y)
            network = Network(x.shape[1])
            optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
            loss_fn = nn.MSELoss()
            best_loss = float("inf")
            best_state: dict[str, torch.Tensor] | None = None
            remaining_patience = config.patience
            losses: list[float] = []
            validation_losses: list[float] = []
            for _epoch in range(config.max_epochs):
                network.train()
                optimizer.zero_grad()
                loss = loss_fn(network(x_tensor[train_indices]), y_tensor[train_indices])
                loss.backward()
                optimizer.step()
                network.eval()
                with torch.no_grad():
                    validation_loss = float(
                        loss_fn(network(x_tensor[valid_indices]), y_tensor[valid_indices]).item()
                    )
                losses.append(float(loss.item()))
                validation_losses.append(validation_loss)
                if validation_loss < best_loss - 1e-6:
                    best_loss = validation_loss
                    best_state = {
                        name: value.detach().clone() for name, value in network.state_dict().items()
                    }
                    remaining_patience = config.patience
                else:
                    remaining_patience -= 1
                    if remaining_patience <= 0:
                        break
            if best_state is not None:
                network.load_state_dict(best_state)
            models[node] = (network, parents, x_mean, x_scale, y_mean, y_scale)
            member_history["mechanisms"][node] = {
                "parents": parents,
                "epochs": len(losses),
                "train_loss": losses,
                "validation_loss": validation_losses,
                "best_validation_loss": best_loss,
            }

        potential: dict[int, np.ndarray] = {}
        for treatment in (0, 1):
            simulated = data.copy()
            simulated["T"] = treatment
            for node in order:
                target = outcome if node == "Y" else node
                fitted_network, parents, x_mean, x_scale, y_mean, y_scale = models[node]
                x = simulated[parents].to_numpy(dtype=np.float32)
                x = np.where(np.isfinite(x), x, x_mean)
                with torch.no_grad():
                    prediction = fitted_network(
                        torch.tensor((x - x_mean) / x_scale)
                    ).numpy()
                simulated[target] = y_mean + y_scale * prediction
            potential[treatment] = simulated[outcome].to_numpy(dtype=float)
        estimates.append(float(np.mean(potential[1] - potential[0])))
        histories.append(member_history)

    lower, upper = (
        np.quantile(estimates, [0.025, 0.975])
        if len(estimates) > 1
        else (estimates[0], estimates[0])
    )
    return NeuralSCMResult(
        graph_id=graph.graph_id,
        outcome=outcome,
        backend=f"PyTorch {torch.__version__}",
        ate=float(np.mean(estimates)),
        interval=(float(lower), float(upper)),
        ensemble_estimates=tuple(estimates),
        histories=tuple(histories),
        config=config,
    )


def fit_neural_scm(
    data: pd.DataFrame,
    graph: GraphSpec,
    outcome: str,
    config: NeuralSCMConfig | None = None,
) -> NeuralSCMResult:
    """Fit a graph-constrained SCM and estimate E[Y(do(T=1))-Y(do(T=0))]."""
    config = config or NeuralSCMConfig()
    try:
        import torch  # noqa: F401
    except ImportError:
        return _fit_sklearn(data, graph, outcome, config)
    return _fit_torch(data, graph, outcome, config)
