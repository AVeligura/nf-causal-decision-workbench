# Отчёт координатору: контрольный этап NF-Causal Decision Workbench v2

Дата: 2026-08-17.

## Итог

Научная логика v2 реализована и прошла контрольный этап. Полный Monte Carlo на
5000 репликаций не запускался. Перед ним требуется решение координатора по
экспериментально вырожденному контрасту `maximum_graph`/`structure_oracle` и
завершение UI-приёмки в среде с Qt/EGL.

## Содержательные изменения

1. Полная процедура использует всю α-траекторию, а не последний срез. Введены
   статусы robust/conditionally robust/switching/pilot/abstain, точки изменения,
   графы-триггеры, диапазон устойчивости и независимое операционное правило.
2. Hard-set вычисляется однократно без μΓ, ранжирования и истории.
3. Истинная и оценочная ценность используют один модуль и одинаковые веса,
   затраты, охват и побочные исходы.
4. EVI пилота основана на 50+ виртуальных пилотных выборках и сохраняет gross,
   net, MCSE, seed и число симуляций.
5. Корпус свидетельств заменён утверждениями уровня рёбер, путей и правил из
   шести типов источников. Прямой `score_graph` воспроизводит
   `G1=0.92, G2=0.81, G3=0.67, G4=0.43`.
6. Пресеты больше не отменяют ручные параметры; редактируются коэффициенты
   ценности и свидетельства, доступен сброс.
7. Исправлены имя CATE-модели, импортный паспорт, overlap-диагностика, семантика
   времени/памяти, replay и PDF-паспорт.

## Тесты и приёмка

- Ruff и compileall: PASS.
- Целевая mypy-проверка 8 изменённых модулей: PASS с новым cache-dir.
- Не-GUI тесты: 56/56 PASS, включая 28 новых тестов научной логики.
- Детерминированные примеры: 6/6 совпали с ожиданиями.
- Эталонный запуск: `run-8adee0cc5217`; checksum PASS.
- Replay: `run-6c3eb54f49e9`; эффекты, решения, α-траектория и пилот совпали.
- Экспорты CSV/XLSX/Parquet/JSON/PDF/PNG/SVG/GraphML: PASS.
- PDF визуально проверен после исправления переполнения таблицы.
- UI/скриншоты: BLOCKED, в контейнере отсутствует `libEGL.so.1`.

## Контрольные примеры

- устойчивое `a2`: PASS;
- устойчивое `a0`: PASS;
- оправданный `a1`: net EVI `0.009413`, внутренняя MCSE `0.000760`, seed `7`;
- переключение по α: `α=0.60`, добавление `G2`, безусловное действие не назначено;
- потеря идентификации: abstain;
- истинная структура вне Γ: `true_graph_id=None`.

## Smoke Monte Carlo

`mc-smoke-v2-20260817-r2`: 100/100 завершено, 10 повторов на ячейку,
`integrity_check=ok`, дубликатов нет.

Истинные optimum: 40 × `a2`, 40 × `a1`, 20 × `a0`. Полная процедура и hard-set
различались по операционному действию в 1% репликаций. Это не демонстрация
преимущества: средний regret обоих равен `0.01573`, у maximum graph и oracle —
`0.00114`. Ошибочный `a2` не наблюдался.

Статусы полной траектории: robust 42%, conditionally robust 46%, pilot 2%,
abstain 10%. Изменение действия по α обнаружено в 45% репликаций.

## Решение, требуемое от координатора

`maximum_graph` и `structure_oracle` совпали в 100% smoke-репликаций. Кодовые
ветви независимы, но DGP задаёт истинный `G1` во всех сценариях внутри Γ, а
maximum graph также всегда `G1`.

Нужно утвердить один вариант:

- сохранить DGP и прямо признать contrast maximum/oracle неинформативным; либо
- согласовать механизм разных истинных структур внутри тех же десяти ячеек,
  обновить предрегистрацию и повторить smoke.

Также требуется выполнить четыре UI-теста и получить три реальные скриншота на
системе с работающей Qt/EGL. До этих решений полный эксперимент не запускать.

## Изменённые файлы

Научное ядро: `src/domain/models.py`, `src/domain/value_model.py`,
`src/engine/decision.py`, `src/engine/evidence.py`, `src/engine/estimation.py`,
`src/engine/identification.py`, `src/engine/passport.py`, `src/engine/pipeline.py`,
`src/engine/stability.py`, `src/study/dgp.py`, `src/study/methods.py`,
`src/study/metrics.py`, `src/study/monte_carlo.py`, `src/study/oracle.py`,
`src/study/scenarios.py`.

Runtime/UI: `src/runtime/import_export.py`, `src/runtime/repository.py`,
`src/ui/main_window.py`, `src/ui/state.py`, `src/ui/workspaces.py`,
`src/visualization/plots.py`.

Контроль: `tests/test_scientific_logic_v2.py`, `scripts/run_control_examples.py`,
`scripts/build_smoke_report.py`, `scripts/run_acceptance_v2.py`,
`scripts/run_reference.py`, документы `docs/*_V2.md`, README, CHANGELOG и
KNOWN_LIMITATIONS.

## Точные каталоги результатов

- Эталон: `artifacts/reference_repository_v2/runs/run-8adee0cc5217/`.
- Replay: `artifacts/reference_repository_v2/runs/run-6c3eb54f49e9/`.
- Smoke: `artifacts/experiments/mc-smoke-v2-20260817-r2/`.
- Контрольные примеры: `artifacts/control_examples/control_examples.json`.
- Приёмка: `artifacts/acceptance_v2/acceptance_report_v2.md`.
- Тесты: `artifacts/non_ui_pytest_v2.xml`.
