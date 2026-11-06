# Материалы статьи (ВАК)

## Файлы

| Файл | Назначение |
|------|-----------|
| `article.tex` | текст статьи (рус., pdflatex, T2A) |
| `references.bib` | библиография (BibTeX; в тексте сейчас `thebibliography` — п. «Переключение на BibTeX») |
| `results_table.tex` | таблица метрик прогона (генерируется `applecounter run/report`) |
| `results_yield.tex` | таблица счёта и урожая по деревьям (генерируется) |
| `figures/` | рисунки из прогона (pipeline, PR, счёт по деревьям и т.д.) |
| `journals_vak.csv` | перечень целевых журналов ВАК (перечень от 15.06.2026) |

## Сборка

```powershell
cd paper
pdflatex -interaction=nonstopmode article.tex
pdflatex -interaction=nonstopmode article.tex   # второй проход: ссылки и citations
```

Статья собирается в `article.pdf` (8 с., рис. и табл. на месте).
Шрифты T2A — из `cm-super` (`tlmgr install cm-super`, если в системе нет).

## Обновление результатов

После нового прогона:

```powershell
# из корня проекта
$env:PYTHONPATH="src"; $env:PYTHONIOENCODING="utf-8"
python -m applecounter run --video <видео> --out outputs\run
python -m applecounter report --run outputs\run     # пересборка отчёта/таблиц
Copy-Item outputs\run\results_table.tex,outputs\run\results_yield.tex paper\
Copy-Item outputs\run\figures\*.png paper\figures\
```

Числа в таблицах статьи обновляются автоматически через `\input{results_table.tex}`
и `\input{results_yield.tex}`; метрики в тексте (раздел 4.2) — вручную.

## Целевые журналы ВАК

Категории (по перечню ВАК): **К1** — ИИ и принятие решений / программная
инженерия / информатика и автоматизация; **К2** — проблемы ИИ; сельскохозяйственные;
**К3** — агробиотехнологии.

```powershell
$env:PYTHONPATH="src"; python -m applecounter journals            # все
$env:PYTHONPATH="src"; python -m applecounter journals --category К1
```

Список (сокращённо):

- **К1**: «Искусственный интеллект и принятие решений» (2071-8594),
  «Программная инженерия» (2220-3397), «Информатика и автоматизация»
  (2713-3192), «Информационно-управляющие системы» (1684-8853).
- **К2**: «Проблемы искусственного интеллекта» (2413-7383),
  «Инженерный вестник Дона» (2073-8633), «Вестник аграрной науки Дона»
  (2075-6704).
- **К3**: «Агробиотехнологии и цифровое земледелие» (2782-490X).

Выбор: для акцента на методе — К1 («Программная инженерия» или
«Информатика и автоматизация»); для акцента на сельскохозяйственной
прикладной задаче — К2/К3.

## Переключение на BibTeX (по требованию журнала)

1. Убрать блок `thebibliography` из `article.tex`, поставить:

```latex
\bibliographystyle{gost2003}   % или unsrt, если стиля нет
\bibliography{references}
```

2. Сборка: `pdflatex → bibtex → pdflatex → pdflatex`.

## Адаптация под журнал

1. Титул: ФИО, уч. степени, место работы, ORCID, e-mail, аннотация/keywords
   (требования разные — сверить с «Инструкцией для авторов» на сайте журнала).
2. Объём: 8–15 с.; таблиц рисунков — не более 5–8 суммарно.
3. Список литературы: многие журналы требуют ссылки на российские издания
   и на сам журнал; RINCY/РИНЦ — проверить индексацию журнала.
4. Подача: через сайт журнала (PDF + сопроводительное письмо), см. `url`
   в `journals_vak.csv`.
