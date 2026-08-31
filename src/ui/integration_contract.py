from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView


FIXED_OUTCOME_TEXT = "Y_CR и Y_CFO · рассчитываются оба"
FIXED_ESTIMAND_TEXT = "ATE · фиксированный основной estimand"
FIXED_VERSION_TEXT = "Полный пакет / без полного исполнения · фиксированный контраст"
FIXED_HORIZON_QUARTERS = 4


def apply_fixed_causal_contract(workspace) -> None:
    """Render the preregistered V3.1 causal contract as information, not controls."""

    workspace.outcome.blockSignals(True)
    workspace.outcome.clear()
    workspace.outcome.addItem(FIXED_OUTCOME_TEXT)
    workspace.outcome.setEnabled(False)
    workspace.outcome.setToolTip(
        "V3.1 рассчитывает два отдельных причинных вопроса: Y_CR и Y_CFO."
    )
    workspace.outcome.blockSignals(False)

    workspace.estimand.blockSignals(True)
    workspace.estimand.clear()
    workspace.estimand.addItem(FIXED_ESTIMAND_TEXT)
    workspace.estimand.setEnabled(False)
    workspace.estimand.setToolTip(
        "Основной estimand V3.1 — ATE. CATE формируется как дополнительная диагностика неоднородности."
    )
    workspace.estimand.blockSignals(False)

    workspace.horizon.blockSignals(True)
    workspace.horizon.setValue(FIXED_HORIZON_QUARTERS)
    workspace.horizon.setEnabled(False)
    workspace.horizon.setToolTip("Горизонт V3.1 зафиксирован: 4 квартала.")
    workspace.horizon.blockSignals(False)

    workspace.version.blockSignals(True)
    workspace.version.clear()
    workspace.version.addItem(FIXED_VERSION_TEXT)
    workspace.version.setEnabled(False)
    workspace.version.setToolTip(
        "Основной анализ использует фиксированный контраст полного пакета относительно режима без полного исполнения; partial учитывается отдельно."
    )
    workspace.version.blockSignals(False)


def configure_evidence_editor(workspace, *, editable: bool) -> None:
    """Keep reference evidence immutable and expose only mathematically active fields."""

    table = workspace.evidence_table
    table.setEnabled(True)
    table.setEditTriggers(
        QAbstractItemView.EditTrigger.DoubleClicked
        if editable
        else QAbstractItemView.EditTrigger.NoEditTriggers
    )

    editable_columns = {4, 5, 6} if editable else set()
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item is None:
                continue
            flags = item.flags()
            if column in editable_columns:
                flags |= Qt.ItemFlag.ItemIsEditable
            else:
                flags &= ~Qt.ItemFlag.ItemIsEditable
            item.setFlags(flags)
