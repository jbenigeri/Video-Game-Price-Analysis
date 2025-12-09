## Checklist: Publish Notebooks with Jupyter Book (MyST)

Jupyter Book v2 is powered by [MyST](https://mystmd.org). The workflow has changed from the old `_toc.yml`/`_config.yml` approach.

### Setup
- [x] Clone `Video-Game-Price-Analysis` repo locally
- [x] `pip install jupyter-book` (or `conda install jupyter-book`)
- [x] `jupyter-book init` (creates `myst.yml`)

### Configure `myst.yml`
- [ ] Edit `myst.yml` to set project metadata:
```yaml
version: 1
project:
  id: 5d23f48c-c4d3-4f07-9929-a845017dd2ee
  title: Video Game Price Analysis
  description: Analyzing Steam game pricing patterns, inflation impact, and COVID-19 effects
  authors:
    - name: Jacob Benigeri
  github: https://github.com/jbenigeri/Video-Game-Price-Analysis
site:
  template: book-theme
```

### Generate Table of Contents
- [ ] Run: `jupyter-book init --write-toc`
  - This auto-generates a `_toc.yml` from your notebooks
- [ ] Or manually create `_toc.yml`:
```yaml
format: jb-book
root: README
chapters:
  - file: src/nb1__exploratory_data_analysis
  - file: src/nb2__prices_analysis
```

### Build Locally
- [ ] `jupyter-book build .`
  - Builds HTML output to `_build/html/`
- [ ] Preview locally: `jupyter-book start`
  - Opens a local server to preview the book

### Deploy to GitHub Pages
- [ ] Install ghp-import: `pip install ghp-import`
- [ ] Deploy: `ghp-import -n -p -f _build/html`
- [ ] GitHub repo → Settings → Pages → Source: `gh-pages` branch

### Verify
- [ ] Visit: https://jbenigeri.github.io/Video-Game-Price-Analysis/

### Link from Website
- [ ] Update `projects.md` in `jbenigeri.github.io` with link to published book

---

## Quick Reference Commands

| Command | Description |
|---------|-------------|
| `jupyter-book init` | Initialize a new MyST project |
| `jupyter-book init --write-toc` | Auto-generate table of contents |
| `jupyter-book build .` | Build the book to `_build/html/` |
| `jupyter-book start` | Start local preview server |
| `jupyter-book clean .` | Remove build artifacts |
