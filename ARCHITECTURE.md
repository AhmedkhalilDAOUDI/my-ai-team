# My AI Team architecture

The browser talks only to the local FastAPI service. FastAPI validates identity, project access, model choices, budgets, uploads, and workflow definitions before calling modular provider adapters. SQLite stores project-scoped application state; API keys remain environment variables and never enter SQLite.

```text
Browser interfaces
  ├─ Debate Engine ────┐
  ├─ Direct Workspace ─┼─ FastAPI ─ Orchestrator ─ Provider adapters
  ├─ Live Chat ────────┤     │              ├─ OpenAI Responses API
  ├─ Builder Workspace ┤     │              ├─ DeepSeek Chat API
  └─ Studio ───────────┘     │              ├─ Isolated Git worktrees
                             │              └─ OpenAI-compatible plugins
                             ├─ Retrieval ─ chunks, embeddings, citations
                             ├─ Graph ───── local entities/edges ─ Neo4j
                             ├─ Job worker ─ evaluations, workflows, n8n
                             └─ SQLite (projects, prompts, knowledge, runs, usage)
```

The Debate Engine is the primary product boundary. It validates two to four participants, preserves each model's identity and owned statements across four stages, routes live moderator instructions between stages, and anonymizes the resulting transcript before one to three independently configured juries score it. Jury reports are aggregated deterministically, while evidence policies, a claim ledger, citation-label auditing, optional baseline comparison, and blind convergence checks make the decision process inspectable. Full transcripts, checkpoints, usage, reproducibility metadata, appeals, and reports are persisted as project-scoped runs; legacy discussion endpoints remain available for compatibility.

Builder Workspace adds a separate mutation boundary: FastAPI creates a task-specific Git branch and worktree, models return validated structured file operations, and a fixed command allowlist performs verification without invoking a shell. The main worktree remains read-only until the user explicitly approves a merge; dirty-main checks prevent accidental mixing with unrelated local changes.

Documents are chunked once and retrieved with lexical ranking or optional OpenAI embeddings. Retrieved chunks keep document/chunk identifiers through generation so answers can cite their evidence. A lightweight local graph is extracted during indexing and can be synchronized to Neo4j without changing the retrieval or provider boundary.

The built-in worker claims durable SQLite jobs and resumes pending work after restart. Project filtering provides practical local user isolation, while owner-only routes protect plugins, global settings, users, and backups. For internet-scale deployment, move identity to an external provider, SQLite to a server database, jobs to a dedicated queue, uploads to object storage, and enforce HTTPS with secure cookies.
