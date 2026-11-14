"""Generate the thin Colab launcher; all training implementation is importable/tested."""
import json
from pathlib import Path
from textwrap import dedent

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": dedent(text).strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": dedent(text).strip().splitlines(keepends=True)})


md("""
    # Text-JEPA v5 · Colab pilot suite

    **Выбрано:** 128M → затем ~300M; контекст 1024; общий батч 128 документов;
    70% FineWeb-Edu + 30% проверенных решений OpenR1; английские пилоты;
    двуязычный EN/RU BPE-токенизатор; infill + suffix; без EMA целевой модели.

    Сравнение: **постоянный SIGReg / плавный SIGReg / span-MLM**, по два seed.
    Вес адаптивного SIGReg ограничен 0.20–0.45, импульсов нет.
    Все численные настройки контроллера — проверяемые гипотезы.

    Выберите GPU runtime (G4, если доступен). В Colab Secrets добавьте `HF_TOKEN`
    с правом записи в свой приватный репозиторий и разрешите доступ этому ноутбуку.
    Поля токенов в коде отсутствуют. Все подготовленные токены и checkpoints
    записываются только в **приватный Hugging Face**.

    Сначала выполняются benchmark и подготовка данных, затем шесть запусков.
    Бюджет по умолчанию **18 часов**, допустимый выбранный диапазон 12–24.
    Реальное время, включая подготовку, оценку и загрузки, нужно контролировать:
    benchmark даёт оценку, а не гарантию. Большая модель автоматически не запускается.
""")
code("""
    import os, sys, subprocess, json, time
    from pathlib import Path
    ROOT = Path('/content/salm-v5')
    BRANCH = 'experiment/text-jepa-v5-colab'
    if not (ROOT / '.git').exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH,
                        'https://github.com/crystalcali/salm.git', str(ROOT)], check=True)
    os.chdir(ROOT)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[text-jepa-v5]'], check=True)
    print('Code revision:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    import torch
    assert torch.cuda.is_available(), 'Выберите GPU runtime и перезапустите среду'
    print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)
    assert torch.cuda.is_bf16_supported(), 'Для этого запуска нужен BF16'
""")
md("""
    ## 1. Приватное хранилище и идентификатор эксперимента

    **При продолжении оставьте тот же `EXPERIMENT_ID`.** Новый ID создаёт новое сравнение.
    План с token budget и microbatch сохраняется на Hub до обучения; повторный запуск
    использует тот же план, подготовленные данные и checkpoints.
""")
code("""
    from google.colab import userdata
    from huggingface_hub import HfApi, hf_hub_download
    from salm.text_jepa_v5.hub import PrivateHub, safe_path
    from salm.text_jepa_v5.config import Config
    from dataclasses import replace, asdict

    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    api = HfApi()
    HF_REPO = api.whoami()['name'] + '/text-jepa-v5-private'
    EXPERIMENT_ID = 'pilot-v5-001'
    safe_path(EXPERIMENT_ID)
    TOTAL_GPU_HOURS = 18  # 12–24; перед новым экспериментом можно изменить
    SEEDS = [42, 43]
    BASE = Config.read('configs/text_jepa_v5/base.json')
    hub = PrivateHub(HF_REPO)
    DATA = Path('/content/text-jepa-v5-data') / EXPERIMENT_ID
    OUT = Path('/content/text-jepa-v5-runs') / EXPERIMENT_ID
    OUT.mkdir(parents=True, exist_ok=True)
    print('Private repository:', HF_REPO, '| Experiment:', EXPERIMENT_ID)
""")
md("""
    ## 2. Измерение скорости и фиксация бюджета

    Измеряются полные шаги обучения, включая GradCache. Выбирается самый быстрый
    успешно измеренный microbatch для каждого метода при неизменном общем батче 128.
    35% времени резервируются под данные, диагностики, оценки и сеть.
    Пилот ограничен 200M входных токенов на запуск. Если расчёт даёт меньше 50M,
    сначала проверьте GPU и бюджет, прежде чем запускать всю серию.
""")
code("""
    from salm.text_jepa_v5.runner import benchmark
    PLAN_PATH = f'plans/{EXPERIMENT_ID}/plan.json'
    if api.file_exists(HF_REPO, PLAN_PATH):
        plan = json.loads(Path(hf_hub_download(HF_REPO, PLAN_PATH)).read_text())
        assert plan['base_fingerprint'] == BASE.fingerprint(), 'Изменён BASE: используйте новый ID'
        SEEDS = plan['seeds']
        TOTAL_GPU_HOURS = plan['total_hours']
        print('Restored existing experiment plan')
    else:
        measured = benchmark(BASE, micro_batches=(8, 16, 32))
        best = {}
        for arm in ('fixed', 'adaptive', 'mlm'):
            valid = [r for r in measured['results'] if r['arm'] == arm and not r.get('oom')]
            assert valid, f'Нет рабочего microbatch для {arm}'
            best[arm] = max(valid, key=lambda r: r['tokens_per_second'])
        total_seconds_per_token = len(SEEDS) * sum(1 / r['tokens_per_second'] for r in best.values())
        estimated_tokens = int(TOTAL_GPU_HOURS * 3600 * .65 / total_seconds_per_token)
        budget = min(200_000_000, estimated_tokens)
        budget = budget // BASE.tokens_per_step * BASE.tokens_per_step
        assert budget >= 50_000_000, 'Меньше 50M на запуск: пересмотрите runtime/бюджет'
        inverse_sum = sum(1 / r['tokens_per_second'] for r in best.values())
        run_caps = {arm: TOTAL_GPU_HOURS * .8 / (len(SEEDS) * inverse_sum * r['tokens_per_second'])
                    for arm, r in best.items()}
        plan = {'base_fingerprint': BASE.fingerprint(), 'token_budget': budget,
                'seeds': SEEDS, 'total_hours': TOTAL_GPU_HOURS, 'benchmark': measured,
                'run_hour_caps': run_caps,
                'micro_batches': {arm: r['micro_batch'] for arm, r in best.items()}}
        api.upload_file(repo_id=HF_REPO, path_or_fileobj=json.dumps(plan, indent=2).encode(),
                        path_in_repo=PLAN_PATH, commit_message='Fix Text-JEPA v5 experiment plan')
    print('Input tokens per run:', f"{plan['token_budget']:,}")
    print('Microbatches:', plan['micro_batches'], '| Seeds:', SEEDS)
""")
md("""
    ## 3. Подготовка одинаковых данных для трёх методов

    Первый запуск обучает BPE на EN/RU и готовит по одному случайному окну из каждого
    документа/семейства задач. В пилотах **все окна обучения английские**.
    Повторный запуск скачивает готовый набор. Проверяются SHA256 и токенизатор.
    Подготовка выполняется на CPU, поэтому в GPU runtime это время тоже расходует квоту.
    Если сессия оборвётся до первого полного сохранения корпуса, подготовку нужно повторить;
    обученный токенизатор и revisions локально переиспользуются.
""")
code("""
    from salm.text_jepa_v5.corpus import prepare, PreparedCorpus
    if api.file_exists(HF_REPO, f'prepared/{EXPERIMENT_ID}/manifest.json'):
        corpus = hub.download_corpus(DATA, EXPERIMENT_ID)
    elif (DATA / 'manifest.json').exists():
        corpus = PreparedCorpus(DATA)
        hub.upload_corpus(DATA, EXPERIMENT_ID)
    else:
        corpus = prepare(DATA, token_budget=plan['token_budget'], seq_len=1024,
                         val_documents=128, tokenizer_documents=10_000,
                         vocab_size=50_000, ru_fraction=0.)
        hub.upload_corpus(DATA, EXPERIMENT_ID)
    print('Corpus fingerprint:', corpus.fingerprint)
    print('Windows:', {name: len(array) for name, array in corpus.arrays.items()})
""")
md("""
    ## 4. Обучение: три метода × два seed

    Конфигурация фиксируется по первому плану. Сначала скачивается checkpoint того же
    запуска; после каждого сохранения он синхронно отправляется на приватный Hub.
    При временной сетевой ошибке ячейка остановится с локальным checkpoint;
    повторите её. Аварийное выключение теряет только шаги после последней успешной загрузки.

    Внутренние retrieval/MSE оцениваются отдельно для infill/suffix и general/reasoning.
    Внешняя оценка: STS validation и SST-2 frozen linear probe (4000 train-примеров).
    Их validation используется для сравнения конфигураций, STS test здесь не запускается.

    Прерывание кнопкой Stop обрабатывается на границе шага с сохранением.
    Это не защищает от мгновенного уничтожения runtime или SIGKILL.
""")
code("""
    from transformers import AutoTokenizer
    from salm.text_jepa_v5.runner import train
    from salm.text_jepa_v5.external import evaluate_external
    import gc

    tokenizer = AutoTokenizer.from_pretrained(DATA / 'tokenizer')
    for seed in SEEDS:
        for arm in ('fixed', 'adaptive', 'mlm'):
            name = f'{arm}_s{seed}'
            folder = OUT / name
            run_hub = PrivateHub(HF_REPO, f'{EXPERIMENT_ID}/{name}')
            if not (folder / 'latest.pt').exists():
                run_hub.download_run(folder)
            cfg = replace(BASE, arm=arm, seed=seed, token_budget=plan['token_budget'],
                          micro_batch=plan['micro_batches'][arm]).validate()
            model, history = train(cfg, corpus, folder, tokenizer.mask_token_id, hub=run_hub,
                                   max_hours=plan['run_hour_caps'][arm])
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if not history or history[-1]['step'] != cfg.steps:
                raise RuntimeError(f'{name} остановлен до конца. При ручной остановке повторите ячейку. '
                    'Если исчерпан run_hour_caps, сначала пересмотрите бюджет; лимит сам не увеличивается.')
            if not (folder / 'external.json').exists():
                scores = evaluate_external(folder / 'latest.pt', corpus)
                print(name, 'STS:', scores['sts_spearman'], 'SST-2:', scores['sst2_accuracy'])
                run_hub.upload_run(folder)
    print('Suite complete')
""")
md("""
    ## 5. Общий отчёт

    Сравнивайте внешнее качество, различия между seed и вычислительные затраты.
    MSE разных JEPA и cross-entropy MLM описывают разные задачи.
    Rank — диагностика распределения; способность к скрытому рассуждению ещё нужно
    отдельно обучать и проверять на следующем этапе.
""")
code("""
    from salm.text_jepa_v5.report import compare
    from IPython.display import display, Markdown, Image
    summary = compare(OUT)
    display(Markdown((OUT / 'comparison.md').read_text()))
    for seed in SEEDS:
        for arm in ('fixed', 'adaptive', 'mlm'):
            image_path = OUT / f'{arm}_s{seed}' / 'training.png'
            if image_path.exists():
                display(Image(filename=str(image_path)))
    api.upload_folder(repo_id=HF_REPO, folder_path=str(OUT),
        path_in_repo=f'reports/{EXPERIMENT_ID}',
        allow_patterns=['comparison.json', 'comparison.md'],
        commit_message='Text-JEPA v5 comparison report')
""")
md("""
    ## Следующий этап: обучаемая модель скрытого рассуждения

    После проверки представлений: предсказание латента следующего блока, внимание
    между блоками и обучаемый выбор **continue / write / tool / finish**.
    Начальный этап — размеченные переходы, затем оценка качества и стоимости цепочки.
    Обучаемая остановка рассуждения не заменяется порогом rank/MSE.
    Это будущая работа; текущий ноутбук проверяет представления и регуляризацию.
""")

notebook = {"cells": cells, "metadata": {
    "colab": {"name": "Text_JEPA_v5_Colab.ipynb"},
    "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
destination = Path(__file__).resolve().parents[1] / "notebooks/text_jepa_v5_colab.ipynb"
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
