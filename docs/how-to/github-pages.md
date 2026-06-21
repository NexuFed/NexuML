# Enable GitHub Pages

NexuML uses [mike](https://github.com/jimporter/mike) to publish versioned documentation to GitHub Pages.

## 1. Enable GitHub Pages in the repository

1. Go to **Settings → Pages** in the NexuML GitHub repository.
2. Set **Source** to **Deploy from a branch**.
3. Set **Branch** to `gh-pages` and directory to `/ (root)`.
4. Click **Save**.

## 2. Trigger the workflow

The `docs.yml` CI workflow deploys automatically:

- **Push to `main`** → deploys the `dev` version
- **Push a `vX.Y.Z` tag** → deploys a versioned release and updates the `latest` alias

## 3. Build locally

Install docs dependencies and serve locally:

```bash
uv sync --extra docs
uv run mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## 4. Build a static site

```bash
uv run mkdocs build --strict
```

The site is written to `site/`.

## 5. Manually deploy a version

```bash
uv run mike deploy --push --update-aliases 1.2.3 latest
uv run mike set-default --push latest
```

## Version switcher

Once deployed, a version switcher appears in the top-right corner of the site (provided by Material for MkDocs + mike integration).
