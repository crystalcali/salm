# SALM Memory-LoRA: генерация с латентной памятью

**Исследовательский модуль для LiquidAI/LFM3-1.7B-Base на NVIDIA DGX Spark.**

Этот модуль входит в репозиторий SALM как **автономный эксперимент**. Он не меняет основной CLI, Streamlit или Trace-JEPA. Эксперименты читают уже подготовленные `runs/trace_data/*_4k.jsonl` и сохраняют артефакты в `runs/memory_lora/` (относительно корня SALM).

## Что реализовано

- **LoRA LFM3**: адаптируются `w1/w2/w3`, attention и conv projections.
- **Past latent memory**: отдельный `no_grad`-проход LFM3 по **прошлому**, до 2048 токенов, захватывает состояние после блока 7, делит на 16 хронологических сегментов и преобразует каждый в латентный слот размерности 256.
- **Memory reader**: cross-attention текущего residual stream к памяти на блоке 16, residual добавление с обучаемым `gate`.
- **Прямое обучение через token CE**: градиент проходит через memory reader, memory projection и LoRA LFM3. **Нет отдельно обучаемого predictor, EMA или stop-gradient по основному token loss**. Состояния старого контекста вычисляются без градиентов ради памяти GPU; это не EMA teacher.
- **Три режима**: `recent_only` (контроль без памяти), `memory` (основной эксперимент), `memory_sigreg` (необязательная регуляризация, которую следует исследовать только при достаточном числе независимых задач в микробатче).
- **Информационно корректный target**: для математики — следующий шаг/абзац; для сообщений агента — следующее сообщение **assistant**, включая вызов инструмента. Полученные ранее `tool`-сообщения допустимы в истории; **будущие** `tool`/`user`-сообщения не используются как target.
- **Анализ памяти**: NLL при настоящей памяти, отключённой памяти и подменённой памятью другого задания. Сохраняются checkpoint, LoRA adapters, memory adapter, optimizer state, JSONL history.

### Научно важное различие

Это **Memory-LoRA**, а **не JEPA в строгом смысле**: первичная версия не содержит обучения латентного предиктора. Так сделано намеренно, чтобы проверить более базовую гипотезу: улучшает ли **латентная память саму генерацию**, обучаясь на token CE. SIGReg — отдельная контролируемая абляция. К JEPA-цели имеет смысл возвращаться только если польза канала памяти будет подтверждена.

### Что не реализовано

- Нет attention-guided разрезов или RL; пока используются границы шагов и агентских сообщений.
- Нет сквозного градиента через полный frozen-backbone проход прошлого (он отключён для экономии памяти); только memory projection/reader и LoRA генератора обучаемы.
- Нет production-сервера с постоянным KV-cache и обновлением памяти при каждом токене: `sample` демонстрирует одну заранее сформированную память и продолжение текущего контекста.
- Нет автоматического исполнения вызовов инструментов во время оценки.
- Нет подтверждённого улучшения внешнего бенчмарка или реально проведённого здесь обучения на GB10.

## Установка из GitHub

1. На DGX Spark обнови SALM после завершения текущего обучения (или в отдельном checkout, если процесс использует каталог проекта):

   ```bash
   cd /home/userdgx1/work/salm/salm
   git pull --ff-only origin main
   source .venv/bin/activate
   export PYTHONPATH="$PWD/experiments/memory_lora_lab${PYTHONPATH:+:$PYTHONPATH}"
   python -m memory_lora.cli print-config
   ```

   Требуются рабочие `torch` (CUDA), `transformers`, `peft`, `safetensors`, `pytest`. Не переустанавливай PyTorch, который уже работает на GB10.

2. Проверка данных, **без загрузки весов LFM3**:

   ```bash
   python -m memory_lora.cli inspect-data --config experiments/memory_lora_lab/configs/memory_pilot.json
   ```

   Проверяй количество `eligible_train_tasks` и `eligible_val_tasks`. Тренер выбирает переходы с достаточной доступной историей; целевые шаги ограничиваются `target_tokens` токенами.

3. **Прежде чем запускать GPU:** дождись завершения текущего 300-step Trace-JEPA, не запускай VLLM одновременно. Предыдущий глобальный OOM уничтожил и VLLM, и Python. Защитный минимум свободной CUDA-памяти по умолчанию — 80 GiB, но это не абсолютная гарантия от OOM.

4. Пилот памяти (10 шагов):

   ```bash
   set -o pipefail
   python -u -m memory_lora.cli train \
     --config experiments/memory_lora_lab/configs/memory_pilot.json \
     2>&1 | tee runs/memory_lora_pilot.log
   echo "Python exit: $?"
   ```

   Если 80 GiB действительно свободно, этот короткий пилот должен пройти без конкурирующих больших моделей. Его GPU-память и производительность нужно измерить на месте — модель здесь не запускалась.

5. Контроль **с теми же шагами, данными и seed, но без памяти**:

   ```bash
   python -u -m memory_lora.cli train \
     --config experiments/memory_lora_lab/configs/recent_only_pilot.json \
     2>&1 | tee runs/recent_only_pilot.log
   ```

6. Проверить контрольные точки (включая **memory zero** и **shuffled memory**):

   ```bash
   python -m memory_lora.cli evaluate \
     --checkpoint runs/memory_lora/memory_pilot/checkpoint_000010 \
     --limit 24
   ```

   Для качественной генерации:

   ```bash
   python -m memory_lora.cli sample \
     --checkpoint runs/memory_lora/memory_pilot/checkpoint_000010 \
     --index 0 --max-new-tokens 120
   ```

   `sample` выводит генерацию при включённой и отключённой памяти. Она не является официальным benchmark и не выполняет инструменты.

7. Режим **memory + SIGReg** запускай только **после** проверки `memory` против `recent_only`. На микробатче 2 SIGReg шумный; нельзя считать улучшение его потерь доказательством лучшей семантики:

   ```bash
   python -u -m memory_lora.cli train \
     --config experiments/memory_lora_lab/configs/memory_sigreg_pilot.json
   ```

8. После удачного пилота можешь создать **новую** конфигурацию для основного опыта (не переписывай уже сохранённую папку пилота):

   ```bash
   python - <<'PY'
   import json
   from pathlib import Path
   p=Path('experiments/memory_lora_lab/configs/memory_pilot.json')
   cfg=json.loads(p.read_text())
   cfg.update({'out':'runs/memory_lora/memory_full_300',
               'max_steps':300,'eval_every':50,
               'checkpoint_every':50,'eval_tasks':50})
   dest=Path('runs/memory_lora/memory_full_300.json')
   dest.parent.mkdir(parents=True,exist_ok=True)
   dest.write_text(json.dumps(cfg,indent=2))
   print(dest)
   PY
   python -u -m memory_lora.cli train --config runs/memory_lora/memory_full_300.json
   ```

   Создай аналогичный отдельный `recent_only_full_300.json`, изменив только `mode` и `out`. Равные шаги не означают равные FLOPs: память требует дополнительного frozen forward.

## Интерпретация результатов

Метрика `memory_gain_nll = val_zero_memory_nll - val_nll` положительна, если **на одном и том же checkpoint** включённая память снижает NLL. Сравни также обученную модель **с памятью** и независимо обученный `recent_only` контроль на одних задачах. Проверка с **перемешанной памятью** ещё строже: если результат не ухудшается, модель, вероятно, не использует индивидуальные сведения задачи.

Осторожно: высокая полезность памяти по teacher-forcing не равнозначна надёжной генерации правильных решений. Проверка реального агентского task success требует отдельной среды и детерминированного исполнителя.

## Безопасность и воспроизводимость

- Результаты обучения хранятся в `runs/` и не должны коммититься. Основной код SALM не меняется этим экспериментальным модулем.
- Не загружать приватные трейсы наружу; модель/токенизатор скачиваются из HF только по стандартному механизму.
- Не перезаписывать существующий run: тренер останавливается при существующем `history.jsonl` без явного resume.
- `--resume` можно указать в новой локальной конфигурации, задав путь к доверенному checkpoint и тот же `out`. Не загружать чужие `.pt` как optimizer state.
- Параметры LoRA, исходный model ID, dataset path, seed и training history сохраняются в run. `history.jsonl` не содержит исходные тексты, но эти данные остаются локальными.
- У GPU-тренинга нет гарантированного результата до первого реального запуска на DGX Spark; CPU юнит-тесты проверяют вычислительную логику и causality.
