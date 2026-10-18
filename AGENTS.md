# Route Resilience Development Guidelines

> [!IMPORTANT]
> - **Preserve Git History**: Avoid rewriting published history — never force push, rebase, or squash commits that have already been published.
> - **Clean Architecture**: Maintain strict separation between `backend/` (FastAPI, PyTorch, graph extraction, disruption/routing) and `frontend/` (React, TanStack Router).
> - **Data & Weights Safety**: Never commit large model weights (`*.pth`), raw imagery (`RouteResilience/images_png/`), or generated analysis output runs (`outputs/`, `backend/outputs/`).
> - **Verification Mandate**: Run pytest suites and frontend builds (`npm run build`) before delivering changes.
