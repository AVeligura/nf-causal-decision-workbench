# Передача V3.1

## 1. Robust EVI

`R0=max_d min_g E_g[V_g(d)]`. После общего результата пилота
`d(z)=argmax_d min_g E_g[V_g(d)|z]`; адаптивная ценность равна
`min_g E_{z|g}[V_g(d(z))]`. Поскольку информацию можно игнорировать,
`R1=max(R0,R1_adaptive)`. Gross EVI — `max(0,R1-R0)`, net EVI — gross EVI за
вычетом information cost. a1 допустимо строго при положительной net EVI.

## 2. Общее свидетельство

Для каждого возможного истинного графа формируется один виртуальный пилотный
контраст `z=(z_CR,z_CFO)`. Этим же наблюдением обновляются все альтернативные
структуры. Независимых псевдонаблюдений по графам нет. Случайные потоки привязаны
к `graph_id`, поэтому перестановка графов результата не меняет.

## 3. Регрессионный пример

При seed 1022: R0=0,0036992322; R1=0,0051610437; gross EVI=0,0014618115;
information cost=0,0002; net EVI=0,0012618115; MCSE=0,0000653182. Инварианты
R1≥R0 и gross EVI=max(0,R1−R0) выполнены.

## 4. Abstain

Abstain остаётся отдельным `decision_status`, но `operational_action=a0`.
Recommendation exact match и operational policy accuracy хранятся отдельно.
Policy value, regret, headline accuracy и erroneous a2 используют только
операционное действие.

## 5. R3.1

Experiment ID `r3-1-20260820`: 444 completed, 0 failed, 2664 строк результатов,
SQLite integrity check `ok`, 444 уникальных design ID и 1332 уникальных seed.
CATE off. Полный эксперимент не запускался. Full operational accuracy 58,56%,
mean regret 0,011424; erroneous a2 — 0 из 268 возможностей.

## 6. Full и hard

Различий статусов: 40 из 444. Различий операционных действий: 0 из 444. Методы
реализовали одну операционную политику; это зафиксировано без подбора параметров.

## 7. Тесты

78 non-GUI тестов прошли; JUnit XML и полный stdout приложены. Mypy: 31 файл,
0 issues. Ruff: all checks passed. Добавлены регрессионные проверки EVI,
R1≥R0, знака net EVI, одного/нескольких графов, перестановки графов,
воспроизводимости, operational accuracy, status/action separation и отсутствия
`inject_result` в новом GUI-runner.

## 8. GUI

Runner V3.1 использует реальные виджеты и QtTest. В текущем Linux-контейнере
фактический запуск BLOCKED из-за отсутствия `libEGL.so.1`; это не засчитано как
PASS. Приложены runner, stdout ошибки, JSON и Markdown со сведениями об окружении.
PNG и фиктивные успешные журналы не создавались.

## 9. Reference/replay

Reference: `run-f247d0fcaf86`. Replay: `run-4368fbb23b7a`. Совпадение в допуске
1e-8, контрольные суммы обоих пакетов корректны.

## 10. Размещение

V3.1 должен быть размещён в новой папке Google Drive рядом с V3. V2, V3 и R3
не перезаписываются. Ссылка фиксируется в итоговом сообщении после загрузки.
