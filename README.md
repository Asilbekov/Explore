# Wet-Ink: ревизия открытых токенов для Nemotron-Labs-Diffusion (training-free)

Реализация идеи **«мокрого чернила»** для `nvidia/Nemotron-Labs-Diffusion-3B`:
диффузионный фронтир открывает токены слева направо, а уже открытые токены в
ограниченном **окне ревизии** позади фронтира перепроверяются, пока фронтир
пишет следующий блок, — если уверенность модели в токене упала ниже порога,
токен возвращается в `[MASK]` и предсказывается заново вместе с «ещё мокрым»
текущим блоком. Чернила у фронтира остаются поправимыми, дальние — высыхают
(покидают окно). **Без какого-либо переобучения.**

Ключевое проектное правило (задано владельцем идеи): **ни одной собственной
формулы, ни одного изменения чужих формул** — только дословное использование
опубликованных механизмов и допускающая проверка совместимости их комбинация.

---

## 1. Что уже умеет сток (и что мы НЕ трогали)

Формальный аппарат модели целиком взят из файла
`modeling_nemotron_labs_diffusion.py`, который поставляется с чекпоинтом
`nvidia/Nemotron-Labs-Diffusion-3B` (HF), и из официального репозитория
[NVlabs/Nemotron-Labs-Diffusion](https://github.com/NVlabs/Nemotron-Labs-Diffusion).

| Механизм | Источник (дословно) | Роль |
|---|---|---|
| Шумовой процесс: `t~U[0,1]` на образец, `p_mask=(1−ε)t+ε`, i.i.d. Бернулли по позициям, шум = только `[MASK]` | `forward_process` | любое подмножество масок — валидное состояние ⇒ ревизия не выводит из распределения |
| Loss: LLaDA-ELBO на масках + `ar_loss_weight·L_AR` | `forward`, `config.json` | joint AR+диффузия (три-модальность) |
| Коммит: Gumbel-argmax `x0`, уверенность `x0_p = softmax(logits)[x0]`, top-k бюджет, порог, top-1 всегда | `_get_transfer_index` | выбор токенов на фиксацию за шаг |
| Бюджет шагов: равномерное разбиение масок по шагам, остаток вперёд | `_get_num_transfer_tokens` (≡ LLaDA `get_num_transfer_tokens`) | «один денойз-шаг на токен» |
| `steps_per_block = block_length`, AR-сеялка (`next_token`), каузальный KV-префикс, bidirectional-денойз блока | `generate()` | «шагов диффузии = открытых токенов» и фронтир — **уже в стоке** |
| Обрезка KV-кэша | `_crop_dynamic_cache` (используется `linear_spec_generate` при откате спекуляции) | хирургия кэша для окна ревизии |
| Порог уверенности `draft_conf >= threshold` (top-1 форсируется) | `linear_spec_generate` | семантика порога коммита |

Из стока следует, что из идеи владельца **уже реализовано в модели**:
«шаг = токен» (`steps_per_block = block_length`), фронтир слева-направо
(semi-AR блоки + каузальный префикс + AR-сеялка), замороженный префикс в кэше.

## 2. Единственная добавка — ревизия открытых токенов

В стоке открытые токены заморожены: в `_get_transfer_index` строка
`confidence = torch.where(mask_index, x0_p, -inf)` даёт открытым позициям
`-inf`, и блок после завершения уезжает в KV-кэш навсегда. Это единственное
реальное ограничение для идеи «пересматривать прошлое».

**Комбинация №1 (одобрена после проверки совместимости):** формула уверенности
LLaDA (`generate.py` L174–183 репозитория ML-GSAI/LLaDA — побуквенно та же,
что в `_get_transfer_index` Nemotron и в `sampler.py::_unmask_low_confidence`
репозитория NVlabs) применяется к **открытым позициям окна ревизии** через
второй дословный вызов того же `_get_transfer_index`: его пороговый фильтр
реализует тест `draft_conf >= threshold`, а позиции, тест не прошедшие
(дополнение возвращённого `transfer_index`), возвращаются в `[MASK]`.

Честная атрибуция: в **ванильном** LLaDA эта формула применяется только к
маскированным позициям (порядок открытия); ревизии открытых токенов там нет.
Распространение той же опубликованной формулы и того же опубликованного
порогового теста на открытые позиции окна — это и есть композиция,
предложенная и согласованная здесь. Её корректность опирается на (а) i.i.d.
Бернулли-маскирование при обучении (любое подмножество масок — валидное
состояние) и (б) выравнивание окна по границам блоков `block_size=32`
(денойз-спан `[хвост·W блоков + текущий блок]` видит чистый префикс ровно как
шумный блок на обучении).

Отвергнуто (по правилу «не изобретать»): uniform-ре-шум (GIDD-гибрид — у
Nemotron в `forward_process` только `[MASK]`, uniform-токены вне распределения
и у GIDD потребовалось переобучение); дистанционно-затухающие расписания
(Rolling Diffusion — непроверенный доменный сдвиг + модификация чужой формулы);
дистиллированные корректоры (ProSeCo/SPEED — требуют обучения). Ближайший
легальный аналог «ближние меняются сильнее» — confidence-критерий: коммит у
фронтира сильнее всего меняет контекст соседних (недавно открытых) токенов,
именно их уверенность и падает.

## 3. Алгоритм (каждая строка — с источником)

```
1. prefill: каузальный forward промпта → KV-кэш, next_token     [сток generate L421–435]
2. для каждого блока b:
   2.1 спан S = [W предыдущих завершённых блоков] + [блок b]    [обобщение block_slice;
        KV-кэш обрезается до начала S                             crop — сток _crop_dynamic_cache]
   2.2 в S: новые позиции → [MASK]; позиция 0 блока ← next_token [сток L438–443]
   2.3 денойз-цикл (шагов ≤ |S|, как в стоке):
        logits = forward(S, cropped_cache, use_cache=False)     [сток L477–481, bidirectional]
        коммит: masked с x0_p ≥ threshold, top-1 всегда          [сток _get_transfer_index — без изменений]
        ремаскинг: ОТКРЫТЫЕ позиции окна с x0_p < threshold      [LLaDA low-confidence — та же формула,
        → обратно в [MASK]                                         тот же порог, без −inf; реализовано
                                                                   вторым дословным вызовом _get_transfer_index]
   2.4 пост-блок: каузальный forward всего S → кэш + next_token  [сток L500–519, спан-расширен]
```

Параметры: `threshold=0.9` (сток-дефолт), окно `revision_blocks ∈ {0,1,2}`
(0 = побитовая эквивалентность стоку), опциональный строгий режим
`commit="fixed1"` — стратегия `fixed` с бюджетом k=1 из `sampler.py` NVlabs.

**Инженерные гарантии (чисто управляющая логика, не формулы):**

- *Safety tail* — после основного цикла, если маски в спане остались (ремаскинг
  может их добавлять), добиваются только коммитами (ремаскинг выключен), чтобы
  восстановить сток-инвариант «ни одного `[MASK]` в KV-кэше». При срабатывании
  сток-остановки по EOS хвост пропускается: сток сам идёт в пост-блок с масками
  (генерация тут же заканчивается, кэш не используется) — побитовая семантика
  стока сохранена.
- *Хард-кап* шагов `4·|S|` — предохранитель от осцилляций (на случай
  «переговорчивой» модели); счётчик срабатываний пишется в статистику.
- *Ремаскинг активен, пока фронтир пишет текущий блок* (в блоке есть маски) —
  «чернила высыхают», когда фронтир ушёл дальше; это же гарантирует завершимость.

## 4. Локальная верификация (CPU, без весов)

`tests/test_smoke.py` собирает крошечную случайную модель **на тех же самых
файлах архитектуры** (`tests/vendor/`, дословные копии с HF, Apache-2.0) и
проверяет:

| Проверка | Результат |
|---|---|
| сток `generate()` работает | ✅ |
| `revision_blocks=0` **побитово идентичен стоку** (ids и NFE), включая путь раннего EOS | ✅ |
| W=1, W=2: без `[MASK]` на выходе, ремаскинг срабатывает, токены в оконе реально ревизуются | ✅ |
| контроль `threshold=0.0`: ремаскинг инертен, safety tail не нужен | ✅ |
| `fixed1`, `causal_context=False`, батч B=2, детерминизм при T=0 | ✅ |
| **кэш-инвариант**: финальный кэш после всех crop+rebuild равен свежему каузальному forward финальной последовательности | ✅ max_abs_err = 0.0 |

Запуск: `pip install torch transformers==5.0.0 && python tests/test_smoke.py`.

## 5. Эксперимент на полноценной модели (Kaggle GPU)

Кёрнел: `wet-ink-nemotron-3b` (приватный, аккаунт Freedomform), полный
протокол в `kaggle/kernel_harness.py`:

1. загрузка полного `nvidia/Nemotron-Labs-Diffusion-3B` (7.7 ГБ), fp16 (fallback fp32);
2. sanity-чат стоком (дефолты чата: `block_length=32, threshold=0.9`);
3. проверка эквивалентности `W=0 ≡ сток` на реальной модели;
4. качественные примеры: сток vs W=1 бок-о-бок + статистика ревизий;
5. **GSM8K A/B** (протокол официального `evaluate.py`: инструкция, chat-template,
   `\boxed{}`/последнее число, NFE): `сток` / `W=1` / `W=2` при `threshold=0.9` +
   абляции порога 0.8/0.95 по бюджету времени; 50 задач, `max_new_tokens=512`.

Результаты: `results/results.json` (обновляется по мере готовности кёрнела).

## 6. Структура

```
wet_ink.py                  # ядро: generate_wet_ink() — единственная точка входа
tests/
  test_smoke.py             # CPU-верификация (см. §4)
  make_tiny_repo.py         # сборка крошечной модели на vendor-файлах
  vendor/                   # дословные файлы архитектуры NVIDIA (Apache-2.0, NOTICE.md)
kaggle/
  kernel_harness.py         # экспериментальный стенд (санити→эквивалентность→GSM8K)
  kernel-metadata.json      # метаданные кёрнела Kaggle
  kernel.py                 # собранный кёрнел (wet_ink.py + kernel_harness.py)
results/                    # результаты кёрнела (results.json, лог)
```

## 7. Воспроизведение

```bash
# локально (CPU, ~1 мин, без весов)
python tests/test_smoke.py

# на GPU (полная модель)
python - <<'PY'
import torch
from transformers import AutoModel, AutoTokenizer
from wet_ink import generate_wet_ink

tok = AutoTokenizer.from_pretrained("nvidia/Nemotron-Labs-Diffusion-3B", trust_remote_code=True)
model = AutoModel.from_pretrained("nvidia/Nemotron-Labs-Diffusion-3B",
                                  trust_remote_code=True).cuda().to(torch.float16).eval()
msgs = [{"role": "user", "content": "Your question"}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
          return_tensors="pt").input_ids.cuda()
out, nfe, stats = generate_wet_ink(model, ids, max_new_tokens=256, block_length=32,
                                   revision_blocks=1, threshold=0.9,
                                   eos_token_id=tok.eos_token_id, return_stats=True)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), stats)
PY
```

Требования: `torch ≥ 2.5`, `transformers == 5.0.0` (файлы модели используют API 5.x:
`masking_utils`, `modeling_layers` и т.д.), GPU ≥ 16 ГБ для 3B в fp16.

## 8. Источники

- Nemotron-Labs-Diffusion: модель `nvidia/Nemotron-Labs-Diffusion-3B` (HF), код
  github.com/NVlabs/Nemotron-Labs-Diffusion (Apache-2.0 / NVIDIA Open Model License для весов)
- LLaDA: Nie et al., «Large Language Diffusion Models», NeurIPS 2025,
  github.com/ML-GSAI/LLaDA (`generate.py`, remasking `low_confidence`)
- Контекст (не вошли в сборку, зафиксированы причины): GIDD (ICML 2025,
  arXiv:2504.12210), Rolling Diffusion (ICML 2024), ProSeCo (arXiv:2602.11590,
  обучаемый корректор), Block Diffusion / BD3-LM, Fast-dLLM, Jump Your Steps
  (arXiv:2410.07761), APD (arXiv:2506.00413)

## 9. Лицензия

Код этого репозитория — MIT. `tests/vendor/` — дословные копии файлов NVIDIA
(Apache-2.0, см. `tests/vendor/NOTICE.md`). Веса модели подчиняются NVIDIA
Open Model License и в репозиторий не входят.
