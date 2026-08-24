from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from domain import AlphaCut, DecisionResult, EffectResult, GraphScore, GraphSpec, StabilityProfile

PALETTE = {
    "green": "#2E7D5B",
    "amber": "#D69E2E",
    "red": "#B84A4A",
    "blue": "#2F6FB0",
    "gray": "#8A939C",
    "graphite": "#334155",
    "light": "#EEF3F7",
}

GRAPH_POSITIONS = {
    "T": (0.0, 0.4),
    "L": (1.1, 1.0),
    "D": (2.2, 0.4),
    "Y": (3.3, 0.4),
}


def _figure(figsize=(7.0, 4.0)) -> Figure:
    sns.set_theme(style="whitegrid", font_scale=0.9)
    return Figure(figsize=figsize, tight_layout=True)


def plot_graphs(graphs: tuple[GraphSpec, ...]) -> Figure:
    figure = _figure((9.2, 5.4))
    axes = figure.subplots(2, 2)
    for axis, graph in zip(axes.ravel(), graphs, strict=True):
        endogenous_edges = [
            edge
            for edge in graph.edges
            if edge[0] in GRAPH_POSITIONS and edge[1] in GRAPH_POSITIONS
        ]
        network = nx.DiGraph()
        network.add_nodes_from(GRAPH_POSITIONS)
        network.add_edges_from(endogenous_edges)
        nx.draw_networkx_nodes(
            network,
            GRAPH_POSITIONS,
            node_color="#DDEAF7",
            edgecolors=PALETTE["blue"],
            node_size=1250,
            linewidths=1.5,
            ax=axis,
        )
        nx.draw_networkx_labels(network, GRAPH_POSITIONS, font_size=11, font_weight="bold", ax=axis)
        nx.draw_networkx_edges(
            network,
            GRAPH_POSITIONS,
            edge_color=PALETTE["graphite"],
            arrows=True,
            arrowsize=18,
            width=1.8,
            connectionstyle="arc3,rad=0.08",
            ax=axis,
        )
        axis.set_title(f"{graph.graph_id}: {graph.description}", fontsize=10, loc="left")
        axis.set_axis_off()
    figure.suptitle("Кандидатные причинные структуры G₁–G₄", fontsize=13, fontweight="bold")
    return figure


def plot_scores(scores: tuple[GraphScore, ...], alpha: float | None = None) -> Figure:
    figure = _figure((7.2, 3.7))
    axis = figure.subplots()
    labels = [score.graph_id for score in scores]
    values = [score.mu for score in scores]
    colors = [
        PALETTE["green"] if alpha is None or value >= alpha else PALETTE["gray"] for value in values
    ]
    bars = axis.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.58)
    for bar, value in zip(bars, values[::-1], strict=True):
        axis.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    if alpha is not None:
        axis.axvline(alpha, color=PALETTE["amber"], linestyle="--", label=f"α={alpha:.2f}")
        axis.legend(loc="lower right")
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Степень совместимости μΓ(G), не вероятность")
    axis.set_title(
        "Совместимость причинных структур со свидетельствами", loc="left", fontweight="bold"
    )
    return figure


def plot_alpha_cascade(cuts: tuple[AlphaCut, ...]) -> Figure:
    nonempty = [cut for cut in cuts if not cut.empty]
    figure = _figure((8.0, max(3.2, 0.43 * len(nonempty) + 1.4)))
    axis = figure.subplots()
    graph_order = ["G1", "G2", "G3", "G4"]
    matrix = np.array(
        [[1 if graph in cut.graph_ids else 0 for graph in graph_order] for cut in nonempty]
    )
    sns.heatmap(
        matrix,
        cmap=sns.color_palette(["#ECEFF2", PALETTE["blue"]], as_cmap=True),
        cbar=False,
        linewidths=1,
        linecolor="white",
        xticklabels=graph_order,
        yticklabels=[f"α={cut.alpha:.2f}" for cut in nonempty],
        ax=axis,
    )
    axis.set_title("Каскад вложенных α-срезов Γα", loc="left", fontweight="bold")
    axis.set_xlabel("Структура включена в срез")
    axis.set_ylabel("")
    return figure


def plot_graph_specific_forest(effects: tuple[EffectResult, ...], outcome: str) -> Figure:
    selected = [effect for effect in effects if effect.outcome == outcome]
    figure = _figure((8.2, 4.2))
    axis = figure.subplots()
    y_positions = np.arange(len(selected))[::-1]
    for y, effect in zip(y_positions, selected, strict=True):
        if effect.status.value == "structural_zero":
            axis.scatter([0], [y], marker="D", s=70, color=PALETTE["blue"], zorder=4)
            axis.text(0.01, y + 0.18, "структурный нуль", fontsize=8, color=PALETTE["blue"])
        elif effect.status.value == "partially_identified" and effect.identified_bounds:
            low, high = effect.identified_bounds
            axis.plot(
                [low, high], [y, y], color=PALETTE["amber"], linewidth=6, solid_capstyle="round"
            )
        elif effect.status.value == "not_identified":
            axis.scatter([0], [y], marker="x", s=80, color=PALETTE["red"], linewidths=2)
            axis.text(0.01, y, "not identified", va="center", fontsize=8, color=PALETTE["red"])
        elif effect.estimate is not None:
            interval = effect.interval or (effect.estimate, effect.estimate)
            axis.errorbar(
                effect.estimate,
                y,
                xerr=[[effect.estimate - interval[0]], [interval[1] - effect.estimate]],
                fmt="o",
                color=PALETTE["green"],
                ecolor=PALETTE["graphite"],
                capsize=4,
            )
    axis.axvline(0, color=PALETTE["gray"], linewidth=1)
    axis.set_yticks(y_positions, [effect.graph_id for effect in selected])
    axis.set_xlabel("Графоспецифический эффект и статистический интервал")
    axis.set_title(f"Причинный вывод по структурам: {outcome}", loc="left", fontweight="bold")
    return figure


def plot_cate_profiles(
    effects: tuple[EffectResult, ...], outcome: str, graph_id: str = "G1"
) -> Figure:
    effect = next(
        (item for item in effects if item.outcome == outcome and item.graph_id == graph_id), None
    )
    figure = _figure((8.0, 4.0))
    axis = figure.subplots()
    if effect is None or not effect.cate_profiles:
        axis.text(0.5, 0.5, "CATE-профили не вычислены", ha="center", va="center")
        axis.set_axis_off()
        return figure
    labels = list(effect.cate_profiles)
    values = [effect.cate_profiles[label] for label in labels]
    axis.barh(labels[::-1], values[::-1], color=PALETTE["blue"])
    axis.axvline(0, color=PALETTE["gray"], linewidth=1)
    axis.set_title(f"CATE-профили ({graph_id}, {outcome})", loc="left", fontweight="bold")
    axis.set_xlabel("Оцененный условный средний эффект")
    return figure


def plot_overlap(data: pd.DataFrame) -> Figure:
    figure = _figure((7.5, 3.8))
    axis = figure.subplots()
    if "propensity_true" not in data:
        axis.text(0.5, 0.5, "Propensity score отсутствует", ha="center", va="center")
        return figure
    for treatment, color, label in ((0, PALETTE["gray"], "T=0"), (1, PALETTE["blue"], "T=1")):
        values = data.loc[data["T"] == treatment, "propensity_true"]
        axis.hist(values, bins=28, density=True, alpha=0.55, color=color, label=label)
    axis.legend()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Propensity score")
    axis.set_title("Практическое перекрытие режимов", loc="left", fontweight="bold")
    return figure


def plot_stability_map(profiles: tuple[StabilityProfile, ...]) -> Figure:
    rows = []
    labels = []
    for profile in profiles:
        for point in profile.points:
            labels.append(f"{profile.outcome} · α={point.alpha:.2f}")
            rows.append(
                [
                    point.uniformly_identified,
                    point.functional_stable,
                    point.sign_stable,
                    point.threshold_stable,
                ]
            )
    figure = _figure((8.2, max(4.5, 0.30 * len(rows) + 1.5)))
    axis = figure.subplots()
    sns.heatmap(
        np.asarray(rows, dtype=int),
        cmap=sns.color_palette([PALETTE["red"], PALETTE["green"]], as_cmap=True),
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["Идентификация", "Функционал", "Знак", "Практический порог"],
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title("Карта α-зависимой устойчивости", loc="left", fontweight="bold")
    return figure


def plot_value_regret(decision: DecisionResult) -> Figure:
    actions = ["a0", "a1", "a2"]
    graph_ids = sorted({graph for action in actions for graph in decision.values.get(action, {})})
    value_matrix = []
    regret_matrix = []
    for action in actions:
        value_row = []
        regret_row = []
        for graph in graph_ids:
            value = decision.values[action].get(graph)
            if isinstance(value, tuple):
                value_row.append((value[0] + value[1]) / 2)
            elif value is None:
                value_row.append(np.nan)
            else:
                value_row.append(float(value))
            regret = decision.regrets[action].get(graph)
            regret_row.append(np.nan if regret is None else float(regret))
        value_matrix.append(value_row)
        regret_matrix.append(regret_row)
    figure = _figure((9.0, 4.1))
    axes = figure.subplots(1, 2)
    sns.heatmap(
        value_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        xticklabels=graph_ids,
        yticklabels=actions,
        ax=axes[0],
    )
    sns.heatmap(
        regret_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=graph_ids,
        yticklabels=actions,
        ax=axes[1],
    )
    axes[0].set_title("Управленческая ценность")
    axes[1].set_title("Сожаление")
    figure.suptitle(
        f"Value / regret при α={decision.alpha:.2f}: {decision.status}",
        fontsize=12,
        fontweight="bold",
    )
    return figure


def plot_monte_carlo_distributions(metrics: pd.DataFrame, metric: str = "ate_error") -> Figure:
    figure = _figure((8.2, 4.2))
    axis = figure.subplots()
    if metrics.empty or metric not in metrics:
        axis.text(0.5, 0.5, "Результаты Monte Carlo отсутствуют", ha="center", va="center")
        return figure
    sns.violinplot(data=metrics, x="method", y=metric, hue="scenario", inner="quart", ax=axis)
    axis.axhline(0, color=PALETTE["gray"], linewidth=1)
    axis.set_title(f"Распределение Monte Carlo: {metric}", loc="left", fontweight="bold")
    axis.tick_params(axis="x", rotation=20)
    return figure


def plot_method_comparison(aggregates: pd.DataFrame, metric: str = "regret") -> Figure:
    figure = _figure((8.4, 4.3))
    axis = figure.subplots()
    if aggregates.empty or metric not in aggregates:
        axis.text(0.5, 0.5, "Агрегаты методов отсутствуют", ha="center", va="center")
        return figure
    sns.pointplot(data=aggregates, x="scenario", y=metric, hue="method", dodge=0.25, ax=axis)
    axis.set_title(f"Сравнение методов по сценариям: {metric}", loc="left", fontweight="bold")
    axis.tick_params(axis="x", rotation=25)
    return figure


def plot_runtime_scaling(metrics: pd.DataFrame) -> Figure:
    figure = _figure((7.5, 4.0))
    axis = figure.subplots()
    if metrics.empty or "sample_size" not in metrics:
        axis.text(0.5, 0.5, "Метрики времени отсутствуют", ha="center", va="center")
        return figure
    if "method_decision_seconds" in metrics:
        sns.lineplot(
            data=metrics,
            x="sample_size",
            y="method_decision_seconds",
            hue="method",
            marker="o",
            ax=axis,
        )
        axis.set_ylabel("Добавочное время построения решения, с")
        title = "Добавочные вычислительные затраты конфигураций"
    elif "replication_runtime_seconds" in metrics:
        collapsed = metrics.groupby("sample_size", as_index=False)[
            "replication_runtime_seconds"
        ].mean()
        sns.lineplot(
            data=collapsed,
            x="sample_size",
            y="replication_runtime_seconds",
            marker="o",
            ax=axis,
        )
        axis.set_ylabel("Полное время репликации, с")
        title = "Полное время репликации (не сравнение методов)"
    else:
        axis.text(0.5, 0.5, "Метрики времени отсутствуют", ha="center", va="center")
        return figure
    axis.set_title(title, loc="left", fontweight="bold")
    return figure


def save_figure(figure: Figure, path: str | Path, *, dpi: int = 300) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    return target
