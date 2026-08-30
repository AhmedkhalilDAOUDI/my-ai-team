# My AI Team

My AI Team is a local AI debate workspace where selected models defend assigned positions, challenge one another, and receive an independent scored verdict. Direct Workspace, Live Chat, Builder, retrieval, and Studio support that central debate experience.

## Quick start

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Add at least one provider API key to `.env`, then open <http://127.0.0.1:8000>. A ChatGPT or DeepSeek consumer subscription does not automatically include API access; API credentials, quotas, and billing are managed separately by each provider.

## Included features

- Structured debates with opening statements, cross-examination, rebuttals, closing positions, and an independent jury.
- Per-debater provider, model, identity, and position selection with strict ownership of previous statements.
- Adversarial, decision, and Socratic formats; live moderator pause/intervention; chronological and side-by-side views.
- Jury scoring for reasoning, evidence, responsiveness, and consistency, plus common ground, disagreements, unresolved questions, and a verdict.
- Saved, replayable, exportable debates with evidence attachments and a conservative pre-run cost estimate.
- Visual workflow builder with create, rename, reorder, remove, and execution-mode controls.
- Agent editor with validated provider-specific model selectors, roles, instructions, peer visibility, and sentence limits.
- True incremental streaming and Stop controls in Debate, Direct Workspace, and Live Chat.
- Workflow range controls: start at a step, stop after a step, or skip selected steps; these also support rerunning or continuing a subset of a workflow.
- Named Live Chat conversations and saved Debate/Direct runs in SQLite.
- Reusable PDF, DOCX, TXT, Markdown, CSV, and JSON documents with bounded shared-context injection.
- Markdown, PDF, and JSON exports for saved runs.
- Model-aware cost estimates, provider-reported token accounting, configurable warnings, and a daily budget.
- Automatic retries with exponential backoff for rate limits and temporary provider failures.
- Run status, errors, usage, cancellation state, and exportable results.
- Optional single-owner access-token protection for network deployments.
- Docker packaging and persistent SQLite storage.
- Hybrid document retrieval with source citations, optional OpenAI embeddings, and automatic knowledge-graph extraction.
- Evaluation suites, durable background jobs, prompt-version history, Neo4j synchronization, and n8n webhooks.
- Provider plugins, full backup/restore, and isolated per-user workspaces with one-time access tokens.
- A Builder Workspace where the Supervisor implements changes, DeepSeek reviews and corrects them, tests run locally, and the user approves the Git merge.

## Builder Workspace

Builder tasks run in separate Git worktrees and branches. The Supervisor receives a bounded snapshot of tracked text files and returns structured file changes; the DeepSeek critic reviews the resulting diff and can make corrective edits. The app then runs one explicitly selected allowlisted test command without a shell.

The interface shows the stage timeline, changed files, test output, and complete diff. Rejected work remains isolated for audit. Approval requires an explicit confirmation and is blocked while the main workspace has uncommitted changes; models never write directly to the current branch.

Builder is intended for trusted local repositories. Review every diff before merging, keep secrets outside tracked files, and use normal Git protection and backups for important projects.

## Studio

Studio contains eight tabs:

1. **Agents** — create agents and assign a provider, model, role, and response contract.
2. **Workflows** — build ordered agent pipelines and choose Respond, Critique, Questions, Audit, or Synthesize for each step.
3. **Knowledge** — save reusable documents. Select them in Debate or Direct Workspace when their contents should be shared with the team.
4. **Evaluations** — create repeatable test cases and run them against a workflow in the background.
5. **Graph** — inspect extracted entities and relationships or synchronize them to Neo4j.
6. **Integrations** — configure provider plugins, create isolated users, connect n8n, and back up or restore the app.
7. **Settings** — inspect provider configuration and change output, embedding, timeout, warning, and budget controls.
8. **Runs** — inspect persisted runs and background jobs, then export results as Markdown, PDF, or JSON.

API keys deliberately remain in `.env`; Studio never writes secrets into SQLite.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

The application is available at <http://127.0.0.1:8000>. The `data` directory is mounted into the container, so conversations, documents, workflows, usage, settings, and run history survive restarts.

## Optional access protection

Set a long random value in `.env` before exposing the service beyond localhost:

```dotenv
APP_ACCESS_TOKEN=replace-with-a-long-random-secret
```

The app then redirects browser users to the local sign-in screen. The owner can create additional users in Studio; each receives a private project and a token shown only once. Data is project-isolated, but this remains a local MVP: use HTTPS, secure cookies, a server database, and a production identity provider before exposing it to unrelated internet users.

## Retrieval, graph, and automation

Every uploaded document is chunked, indexed, and given stable citations such as `D3C7`. Keyword retrieval works locally with no extra cost. Semantic embeddings are opt-in under Studio Settings and use `EMBEDDING_MODEL`; enabling them makes separately billed provider API calls.

Set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` to enable graph synchronization. n8n can queue a workflow through `POST /api/webhooks/n8n/workflows/{workflow_id}` and monitor it through `GET /api/jobs`. Jobs survive application restarts.

Provider plugins use an OpenAI-compatible HTTPS endpoint, an environment-variable name for the API key, and an explicit model list. Secrets stay in `.env`; only plugin metadata is stored in SQLite.

## Configuration and billing

Model defaults and provider pricing are configured in `.env`. Runtime limits changed in Studio are persisted in SQLite and override the corresponding non-secret defaults. Usage records prefer provider-reported input and output tokens; interrupted responses and providers that omit usage are marked as estimates.

Pricing is model-aware but remains a local estimate. Update the per-million-token rates when provider pricing changes, and verify invoices in the provider console for authoritative billing.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests never call paid APIs. They cover the structured debate contract, participant validation, verdict persistence, interfaces, CRUD, streaming, retrieval and citations, graph extraction, evaluations, durable jobs, plugins, backups, Builder isolation and merge behavior, settings, costs, and exports.

## Architecture and extension points

See [ARCHITECTURE.md](ARCHITECTURE.md). FastAPI owns validation and orchestration, SQLite stores local state, and small provider adapters isolate external APIs.

FastAPI API documentation is available at <http://127.0.0.1:8000/docs>. Useful integration endpoints include `POST /api/debate/stream`, `POST /api/debate/{run_id}/control`, `POST /api/workflows/{id}/stream`, and `POST /api/chat/stream`.
