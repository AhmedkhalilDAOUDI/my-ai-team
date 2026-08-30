# My AI Team architecture

The browser talks only to the local FastAPI service. FastAPI validates identity, project access, model choices, budgets, uploads, and workflow definitions before calling modular provider adapters. SQLite stores project-scoped application state; API keys remain environment variables and never enter SQLite.

```text
Browser interfaces
  ├─ Direct Workspace ─┐
  ├─ Team Discussion ──┼─ FastAPI ─ Orchestrator ─ Provider adapters
  ├─ Live Chat ────────┤     │              ├─ OpenAI Responses API
  └─ Studio ───────────┘     │              ├─ DeepSeek Chat API
                             │              └─ OpenAI-compatible plugins
                             ├─ Retrieval ─ chunks, embeddings, citations
                             ├─ Graph ───── local entities/edges ─ Neo4j
                             ├─ Job worker ─ evaluations, workflows, n8n
                             └─ SQLite (projects, prompts, knowledge, runs, usage)
```

Documents are chunked once and retrieved with lexical ranking or optional OpenAI embeddings. Retrieved chunks keep document/chunk identifiers through generation so answers can cite their evidence. A lightweight local graph is extracted during indexing and can be synchronized to Neo4j without changing the retrieval or provider boundary.

The built-in worker claims durable SQLite jobs and resumes pending work after restart. Project filtering provides practical local user isolation, while owner-only routes protect plugins, global settings, users, and backups. For internet-scale deployment, move identity to an external provider, SQLite to a server database, jobs to a dedicated queue, uploads to object storage, and enforce HTTPS with secure cookies.
