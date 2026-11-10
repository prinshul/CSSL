# Paper

```bash
python gen_numbers.py
python make_figures.py
latexmk -pdf main.tex
python video/make_video.py
```

`numbers.json` holds all results used in the text, tables, figures and video. From the repository root,
`python tools/export_paper_numbers.py --exp exp --numbers paper/numbers.json --write` refreshes it from the runs.
`python tools/qualitative.py --exp exp --out paper/figures` renders the qualitative figures from the checkpoints;
the qualitative section is included in the PDF once these figures exist.
