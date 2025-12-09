## Checklist: Publish Notebooks with Jupyter Book (MyST)

Jupyter Book v2 is powered by [MyST](https://mystmd.org). 

### Setup
- [x] `pip install jupyter-book` (or `conda install jupyter-book`)
- [x] `jupyter-book init` → creates `myst.yml`

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

### Preview Locally
- [x] `jupyter-book start` → Live dev server at http://localhost:3000
- [x] Or: `jupyter-book build --html` → Builds site AND starts preview server

### Build for Deployment
- [ ] `jupyter-book build --html`
  - Builds static HTML to `_build/html/`
  - Press `Ctrl+C` to stop the preview server after verifying

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
| `jupyter-book init` | Initialize a new MyST project (creates myst.yml) |
| `jupyter-book start` | Start live dev server with hot reload |
| `jupyter-book build --html` | Build static HTML + start preview |
| `jupyter-book build --pdf` | Build PDF export |
| `jupyter-book clean .` | Remove build artifacts |
| `ghp-import -n -p -f _build/html` | Deploy to GitHub Pages |

---

## Troubleshooting

**`jupyter-book build .` fails?**
→ Use `jupyter-book build --html` instead (need to specify format)

**Old commands don't work?**
→ Jupyter Book v2 uses MyST. Old commands like `jupyter-book create book` are deprecated.

**TOC not showing correctly?**
→ Edit the `toc:` section in `myst.yml` to control which files appear
