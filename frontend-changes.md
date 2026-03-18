# Frontend Changes

## Code Quality Tooling

### What was added

**Prettier** — automatic code formatter for HTML, CSS, and JavaScript (the frontend equivalent of Black).

### New files

| File | Purpose |
|------|---------|
| `frontend/package.json` | npm project config; defines `format`, `format:check`, and `lint` scripts |
| `frontend/.prettierrc` | Prettier configuration (100-char line width, single quotes, LF endings, 2-space indent) |
| `frontend/.prettierignore` | Excludes `node_modules/` from formatting |
| `frontend/node_modules/` | Installed Prettier dependency (not committed) |
| `check-frontend.sh` | Root-level script that runs the full frontend quality check suite |

### Formatting applied

Prettier was run against all three frontend source files. Key normalizations:

- **index.html**: lowercased `<!doctype html>`, self-closing void elements (`<meta />`, `<link />`), consistent 2-space indentation throughout
- **style.css**: uniform spacing around selectors, properties, and values
- **script.js**: consistent single quotes, trailing commas in multi-line structures, removed extra blank lines

### How to use

```bash
# Check formatting (CI / pre-commit)
./check-frontend.sh

# Auto-fix formatting
cd frontend && npm run format

# Check only (no writes)
cd frontend && npm run format:check
```
