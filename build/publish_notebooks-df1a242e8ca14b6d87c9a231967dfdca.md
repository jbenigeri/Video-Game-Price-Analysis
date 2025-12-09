## Checklist: Publish 2 Notebooks with Jupyter Book

### Setup
- [ ] Clone `Video-Game-Price-Analysis` repo locally
- [ ] `pip install jupyter-book ghp-import`

### Create Book Structure
- [ ] `cd` into repo root
- [ ] `jupyter-book create book`
- [ ] Copy both notebooks into `book/`:
  - [ ] `cp src/nb1__exploratory_data_analysis_formatted.ipynb book/`
  - [ ] `cp src/src/nb2__prices_analysis.ipynb book/`

### Configure Table of Contents (`book/_toc.yml`)
- [ ] Edit `book/_toc.yml`:
```yaml
format: jb-book
root: intro
chapters:
  - file: nb1__exploratory_data_analysis_formatted
  - file: <second_notebook>
```

### Configure Book Settings (`book/_config.yml`)
- [ ] Edit `book/_config.yml`:
```yaml
title: Video Game Price Analysis
author: Jacob Benigeri
```

### Build
- [ ] `jupyter-book build book/`
- [ ] Preview: open `book/_build/html/index.html` in browser

### Deploy
- [ ] `ghp-import -n -p -f book/_build/html`
- [ ] GitHub repo → Settings → Pages → Source: `gh-pages` branch

### Verify
- [ ] Visit published URL (shown in GitHub Pages settings)

### Link from Website
- [ ] Update `projects.md` in `jbenigeri.github.io` with link to notebook
