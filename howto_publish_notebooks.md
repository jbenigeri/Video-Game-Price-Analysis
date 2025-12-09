## Checklist: Publish Notebooks with Jupyter Book (MyST)

Jupyter Book v2 is powered by [MyST](https://mystmd.org). 

### Setup
- [x] `pip install jupyter-book` (or `conda install jupyter-book`)
- [x] `jupyter-book init` → creates `myst.yml`

### Configure `myst.yml`
- [x] Edit `myst.yml` to set project metadata (title, description, authors, TOC)

### Preview Locally
- [x] `jupyter-book start` → Live dev server at http://localhost:3000
- [x] Or: `jupyter-book build --html` → Builds site AND starts preview server

### Deploy to GitHub Pages (GitHub Actions)

**This is the recommended approach** - uses GitHub Actions to auto-deploy on every push.

1. **Run the setup command:**
   ```bash
   jupyter-book init --gh-pages
   ```
   - Choose branch: `main`
   - Choose action name: `deploy.yml`

2. **Commit and push the workflow:**
   ```bash
   git add .
   git commit -m "Add GitHub Pages deployment workflow"
   git push
   ```

3. **Enable GitHub Pages:**
   - Go to: `https://github.com/<username>/<repo>/settings/pages`
   - Set Source: **GitHub Actions**

4. **Wait for deployment:**
   - Check progress: `https://github.com/<username>/<repo>/actions`
   - Site will be live at: `https://<username>.github.io/<repo>/`

### Verify
- [x] Visit: https://jbenigeri.github.io/Video-Game-Price-Analysis/

---

## Quick Reference Commands

| Command | Description |
|---------|-------------|
| `jupyter-book init` | Initialize a new MyST project (creates myst.yml) |
| `jupyter-book init --gh-pages` | Create GitHub Actions workflow for auto-deployment |
| `jupyter-book start` | Start live dev server with hot reload |
| `jupyter-book build --html` | Build static HTML + start preview |
| `jupyter-book clean .` | Remove build artifacts |

---

## How It Works

The `jupyter-book init --gh-pages` command creates `.github/workflows/deploy.yml` which:
- Triggers on every push to `main`
- Automatically sets `BASE_URL` for GitHub Pages
- Builds the site with MyST
- Deploys to GitHub Pages

No manual `ghp-import` needed - just push to main and the site updates automatically! 🚀

---

## Troubleshooting

**Site shows "BASE_URL configuration" error?**
→ The GitHub Action handles this automatically. Make sure you used `jupyter-book init --gh-pages`.

**Changes not appearing on site?**
→ Check GitHub Actions tab for build status. May take 2-3 minutes to deploy.

**`myst` command not found?**
→ Use `jupyter-book` instead (it wraps myst), or install with `npm install -g mystmd`.

**Local preview doesn't work?**
→ Use `jupyter-book start` for local development (no BASE_URL needed locally).
