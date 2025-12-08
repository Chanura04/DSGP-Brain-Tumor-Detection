# Contributing Guidelines

---

## 🌿 Branching Strategy

- `master`: stable, production-ready code
- `dev`: integrated development branch
- `feature/...`: new features

### Example:
- feature-mri-feature-extractor
- feature-ct-feature-extractor
- feature-fusion-engineer
- feature-classification-engineer

## 📝 Commit Message Convention

### Use the format:
- feat: added feature
- fix: fix bug
- docs: update documentation
- refactor: improve existing code
- test: add tests

## 🔍 Pull Requests

- Create PRs into `dev` branch
- One other member or leader must review before merging
- PR must pass tests (CI)

## 🧹 Code Style

- Python formatting: **Black**
- Linting: **Flake8**
- Testing: **pytest**
- Type checking: **mypy** (optional)

## 📦 Data Handling

- Do NOT commit `data/raw` manually
- Use DVC or shared drive if dataset is large