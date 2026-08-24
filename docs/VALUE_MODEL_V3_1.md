# Ценностная модель V3.1 и двухэтапный пилот a1

Три действия сравниваются на одном двухстадийном горизонте. a0 означает отказ от
воздействия; a2 — полный охват и полную программную стоимость; a1 — пилот на доле
`pilot_share`, наблюдение новых данных и выбор stop/rollout для остатка. При
rollout дополнительная стоимость равна `program_cost_a2-program_cost_a1`.

Полноохватная полезность до программной стоимости имеет вид

`U = multiplier × [ΔCFO + wCR ΔCR + wFC ΔFC + wAR ΔAR − wSales LossSales − wzombie Pzombie]`.

Immediate value a1 равна `pilot_share × U-program_cost_a1`; continuation value
rollout — `(1-pilot_share) × U-(program_cost_a2-program_cost_a1)`; stop имеет
нулевую continuation value. Truth и estimate вызывают один вычислитель и
используют одинаковые единицы, охват и побочные параметры.

В V3.1 preposterior EVI определена через `R0=max_d min_g E_g[V_g(d)]` и
`R1=min_g E_{z|g}[V_g(d(z))]`, где один результат пилота `z` обновляет все
структурные ветви. Доступна стратегия игнорирования информации, поэтому
фактически `R1=max(R0,R1_adaptive)`. Gross EVI равна `max(0,R1-R0)`, net EVI —
gross EVI за вычетом стоимости информации. a1 допустимо строго при `net EVI>0`.

В журнале сохраняются immediate и continuation value по графам, R0, R1,
адаптивное R1 до выбора стратегии игнорирования, gross/net EVI, information
cost, внутренняя MCSE, seed, число виртуальных выборок и способ формирования
общего свидетельства. Подробное определение и контрольные числа находятся в
`FORMAL_ROBUST_EVI_V3_1.md`.
