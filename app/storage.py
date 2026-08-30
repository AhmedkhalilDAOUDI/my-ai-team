import json
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "my_ai_team.db"
CURRENT_PROJECT_ID: ContextVar[int] = ContextVar("current_project_id", default=1)


def current_project_id() -> int: return CURRENT_PROJECT_ID.get()


class Store:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self):
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
              id INTEGER PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS agents (
              id INTEGER PRIMARY KEY, name TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
              role TEXT NOT NULL, instructions TEXT NOT NULL DEFAULT '', max_sentences INTEGER NOT NULL DEFAULT 5,
              can_read_peers INTEGER NOT NULL DEFAULT 1, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS workflows (
              id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS workflow_steps (
              id INTEGER PRIMARY KEY, workflow_id INTEGER NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
              agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE RESTRICT, position INTEGER NOT NULL,
              mode TEXT NOT NULL DEFAULT 'respond', UNIQUE(workflow_id, position)
            );
            CREATE TABLE IF NOT EXISTS conversations (
              id INTEGER PRIMARY KEY, title TEXT NOT NULL, interface TEXT NOT NULL DEFAULT 'chat',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS conversation_messages (
              id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
              speaker TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS usage_records (
              id INTEGER PRIMARY KEY, conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
              provider TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0,
              output_tokens INTEGER NOT NULL DEFAULT 0, estimated_cost_usd REAL NOT NULL DEFAULT 0,
              estimated INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS documents (
              id INTEGER PRIMARY KEY, filename TEXT NOT NULL, content TEXT NOT NULL, characters INTEGER NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY, interface TEXT NOT NULL, workflow_id INTEGER REFERENCES workflows(id) ON DELETE SET NULL,
              prompt TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'running', result_json TEXT NOT NULL DEFAULT '{}',
              error TEXT NOT NULL DEFAULT '', cancel_requested INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS app_settings (
              key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """)
            db.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (1)")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(conversations)")}
            if "openai_model" not in columns:
                db.execute("ALTER TABLE conversations ADD COLUMN openai_model TEXT")
            if "deepseek_model" not in columns:
                db.execute("ALTER TABLE conversations ADD COLUMN deepseek_model TEXT")
            if db.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0:
                self._seed(db)

    def _seed(self, db):
        project_id = db.execute("INSERT INTO projects(name, description) VALUES (?, ?)", ("My Thesis", "Default research workspace")).lastrowid
        agents = [
            ("Supervisor", "openai", "gpt-5.6-sol", "Supervisor and technical builder", "Answer first, decide, and propose the concrete solution.", 5, 1, 1),
            ("DeepSeek Critic", "deepseek", "deepseek-v4-pro", "Adversarial critic and reviewer", "Review the Supervisor answer; identify flaws and corrections.", 5, 1, 1),
            ("Jury", "openai", "gpt-5.6-sol", "Question-only jury", "Ask only decisive questions based on prior answers.", 5, 1, 1),
            ("Completeness Auditor", "deepseek", "deepseek-v4-pro", "Completeness auditor", "Identify important omissions without rewriting prior work.", 5, 1, 1),
        ]
        ids = [db.execute("INSERT INTO agents(name,provider,model,role,instructions,max_sentences,can_read_peers,enabled) VALUES (?,?,?,?,?,?,?,?)", row).lastrowid for row in agents]
        workflow_id = db.execute("INSERT INTO workflows(project_id,name,description) VALUES (?,?,?)", (project_id, "Supervisor Review Pipeline", "Supervisor → Critic → Jury → Completeness Auditor")).lastrowid
        for position, agent_id in enumerate(ids, 1):
            db.execute("INSERT INTO workflow_steps(workflow_id,agent_id,position,mode) VALUES (?,?,?,?)", (workflow_id, agent_id, position, ("respond", "critique", "questions", "audit")[position-1]))

    def list_rows(self, table: str):
        if table not in {"projects", "agents", "workflows"}: raise ValueError("Invalid table")
        with self.connect() as db:
            if table in {"projects","agents","workflows"}:
                return [dict(row) for row in db.execute(f"SELECT * FROM {table} WHERE id=? ORDER BY id" if table=="projects" else f"SELECT * FROM {table} WHERE project_id=? ORDER BY id", (current_project_id(),))]
            return []

    def create(self, table: str, data: dict):
        allowed = {
            "projects": ("name", "description"),
            "agents": ("project_id", "name", "provider", "model", "role", "instructions", "max_sentences", "can_read_peers", "enabled"),
            "workflows": ("project_id", "name", "description"),
        }[table]
        values = {key: data[key] for key in allowed if key in data}
        if table in {"agents","workflows"}: values["project_id"] = current_project_id()
        with self.connect() as db:
            cursor = db.execute(f"INSERT INTO {table} ({','.join(values)}) VALUES ({','.join('?' for _ in values)})", tuple(values.values()))
            return dict(db.execute(f"SELECT * FROM {table} WHERE id=?", (cursor.lastrowid,)).fetchone())

    def update_agent(self, agent_id: int, data: dict):
        allowed = {"name", "provider", "model", "role", "instructions", "max_sentences", "can_read_peers", "enabled"}
        values = {key: value for key, value in data.items() if key in allowed}
        if not values: return None
        with self.connect() as db:
            db.execute(f"UPDATE agents SET {','.join(f'{key}=?' for key in values)} WHERE id=? AND project_id=?", (*values.values(), agent_id,current_project_id()))
            row = db.execute("SELECT * FROM agents WHERE id=? AND project_id=?", (agent_id,current_project_id())).fetchone()
            return dict(row) if row else None

    def delete_agent(self, agent_id: int) -> bool:
        with self.connect() as db:
            try:
                cursor = db.execute("DELETE FROM agents WHERE id=? AND project_id=?", (agent_id,current_project_id()))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Agent is used by a workflow and cannot be deleted.") from exc
            return cursor.rowcount > 0

    def workflow_detail(self, workflow_id: int):
        with self.connect() as db:
            workflow = db.execute("SELECT * FROM workflows WHERE id=? AND project_id=?", (workflow_id,current_project_id())).fetchone()
            if not workflow: return None
            steps = db.execute("""SELECT ws.id,ws.position,ws.mode,a.id agent_id,a.name agent_name,a.provider,a.model,a.role,
              a.instructions,a.max_sentences,a.can_read_peers,a.enabled
              FROM workflow_steps ws JOIN agents a ON a.id=ws.agent_id WHERE ws.workflow_id=? ORDER BY ws.position""", (workflow_id,)).fetchall()
            return {**dict(workflow), "steps": [dict(row) for row in steps]}

    def replace_steps(self, workflow_id: int, steps: list[dict]):
        with self.connect() as db:
            agent_ids = [step["agent_id"] for step in steps]
            if agent_ids:
                placeholders = ",".join("?" for _ in agent_ids)
                found = db.execute(f"SELECT COUNT(*) FROM agents WHERE id IN ({placeholders})", agent_ids).fetchone()[0]
                if found != len(set(agent_ids)):
                    raise ValueError("One or more workflow agents do not exist.")
            if not db.execute("SELECT 1 FROM workflows WHERE id=? AND project_id=?",(workflow_id,current_project_id())).fetchone(): raise ValueError("Workflow not found in this project")
            db.execute("DELETE FROM workflow_steps WHERE workflow_id=?", (workflow_id,))
            for position, step in enumerate(steps, 1):
                db.execute("INSERT INTO workflow_steps(workflow_id,agent_id,position,mode) VALUES (?,?,?,?)", (workflow_id, step["agent_id"], position, step.get("mode", "respond")))
        return self.workflow_detail(workflow_id)

    def delete_workflow(self, workflow_id: int) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM workflows WHERE id=? AND project_id=?", (workflow_id,current_project_id())).rowcount > 0

    def update_workflow(self, workflow_id: int, name: str, description: str):
        with self.connect() as db:
            cursor = db.execute("UPDATE workflows SET name=?,description=? WHERE id=? AND project_id=?", (name, description, workflow_id,current_project_id()))
            row = db.execute("SELECT * FROM workflows WHERE id=? AND project_id=?", (workflow_id,current_project_id())).fetchone()
            return dict(row) if cursor.rowcount and row else None

    def list_conversations(self, interface: str = "chat"):
        with self.connect() as db:
            rows = db.execute("""SELECT c.*, COUNT(m.id) message_count
              FROM conversations c LEFT JOIN conversation_messages m ON m.conversation_id=c.id
              WHERE c.interface=? AND c.project_id=? GROUP BY c.id ORDER BY c.updated_at DESC,c.id DESC""", (interface,current_project_id())).fetchall()
            return [dict(row) for row in rows]

    def create_conversation(self, title: str, interface: str = "chat"):
        with self.connect() as db:
            cursor = db.execute("INSERT INTO conversations(title,interface,project_id) VALUES (?,?,?)", (title, interface,current_project_id()))
            return self._conversation_detail(db, cursor.lastrowid)

    def conversation_detail(self, conversation_id: int):
        with self.connect() as db:
            return self._conversation_detail(db, conversation_id)

    @staticmethod
    def _conversation_detail(db, conversation_id: int):
        conversation = db.execute("SELECT * FROM conversations WHERE id=? AND project_id=?", (conversation_id,current_project_id())).fetchone()
        if not conversation: return None
        messages = db.execute("SELECT * FROM conversation_messages WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
        return {**dict(conversation), "messages": [dict(row) for row in messages]}

    def rename_conversation(self, conversation_id: int, title: str):
        with self.connect() as db:
            cursor = db.execute("UPDATE conversations SET title=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=?", (title, conversation_id,current_project_id()))
            return self._conversation_detail(db, conversation_id) if cursor.rowcount else None

    def update_conversation_models(self, conversation_id: int, openai_model: str, deepseek_model: str):
        with self.connect() as db:
            cursor = db.execute("""UPDATE conversations SET openai_model=?,deepseek_model=?,updated_at=CURRENT_TIMESTAMP
              WHERE id=? AND project_id=?""", (openai_model, deepseek_model, conversation_id,current_project_id()))
            return self._conversation_detail(db, conversation_id) if cursor.rowcount else None

    def delete_conversation(self, conversation_id: int) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM conversations WHERE id=? AND project_id=?", (conversation_id,current_project_id())).rowcount > 0

    def add_conversation_message(self, conversation_id: int, speaker: str, content: str):
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM conversations WHERE id=? AND project_id=?", (conversation_id,current_project_id())).fetchone(): return None
            cursor = db.execute("INSERT INTO conversation_messages(conversation_id,speaker,content) VALUES (?,?,?)", (conversation_id, speaker, content))
            db.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (conversation_id,))
            return dict(db.execute("SELECT * FROM conversation_messages WHERE id=?", (cursor.lastrowid,)).fetchone())

    def clear_conversation_messages(self, conversation_id: int) -> bool:
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM conversations WHERE id=? AND project_id=?", (conversation_id,current_project_id())).fetchone(): return False
            db.execute("DELETE FROM conversation_messages WHERE conversation_id=?", (conversation_id,))
            db.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (conversation_id,))
            return True

    def record_usage(self, conversation_id: int | None, provider: str, model: str, input_tokens: int, output_tokens: int, estimated_cost_usd: float, estimated: bool = False):
        with self.connect() as db:
            cursor = db.execute("""INSERT INTO usage_records
              (conversation_id,provider,model,input_tokens,output_tokens,estimated_cost_usd,estimated)
              VALUES (?,?,?,?,?,?,?)""", (conversation_id, provider, model, input_tokens, output_tokens, estimated_cost_usd, int(estimated)))
            db.execute("UPDATE usage_records SET project_id=? WHERE id=?",(current_project_id(),cursor.lastrowid))
            return dict(db.execute("SELECT * FROM usage_records WHERE id=?", (cursor.lastrowid,)).fetchone())

    def usage_summary(self, days: int = 30):
        with self.connect() as db:
            totals = db.execute("""SELECT COALESCE(SUM(input_tokens),0) input_tokens,
              COALESCE(SUM(output_tokens),0) output_tokens,COALESCE(SUM(estimated_cost_usd),0) cost_usd,
              COUNT(*) request_count FROM usage_records WHERE created_at>=datetime('now',?) AND project_id=?""", (f"-{days} days",current_project_id())).fetchone()
            today = db.execute("""SELECT COALESCE(SUM(estimated_cost_usd),0) cost_usd,COUNT(*) request_count
              FROM usage_records WHERE date(created_at)=date('now') AND project_id=?""",(current_project_id(),)).fetchone()
            providers = db.execute("""SELECT provider,model,SUM(input_tokens) input_tokens,SUM(output_tokens) output_tokens,
              SUM(estimated_cost_usd) cost_usd,COUNT(*) request_count FROM usage_records
              WHERE created_at>=datetime('now',?) AND project_id=? GROUP BY provider,model ORDER BY cost_usd DESC""", (f"-{days} days",current_project_id())).fetchall()
            return {"days": days, "today": dict(today), "total": dict(totals), "providers": [dict(row) for row in providers]}

    def add_document(self, filename: str, content: str):
        with self.connect() as db:
            cursor = db.execute("INSERT INTO documents(filename,content,characters,project_id) VALUES (?,?,?,?)", (filename, content, len(content),current_project_id()))
            return dict(db.execute("SELECT id,filename,characters,created_at FROM documents WHERE id=?", (cursor.lastrowid,)).fetchone())

    def list_documents(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT id,filename,characters,created_at FROM documents WHERE project_id=? ORDER BY id DESC",(current_project_id(),))]

    def document_context(self, document_ids: list[int] | None = None, limit: int = 40_000):
        with self.connect() as db:
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                rows = db.execute(f"SELECT id,filename,content FROM documents WHERE project_id=? AND id IN ({placeholders}) ORDER BY id", (current_project_id(),*document_ids)).fetchall()
            else:
                rows = db.execute("SELECT id,filename,content FROM documents WHERE project_id=? ORDER BY id DESC LIMIT 10",(current_project_id(),)).fetchall()
            parts, used = [], 0
            for row in rows:
                block = f"[Document {row['id']}: {row['filename']}]\n{row['content']}"
                if used + len(block) > limit: block = block[:max(0, limit-used)]
                if block: parts.append(block); used += len(block)
                if used >= limit: break
            return "\n\n".join(parts)

    def delete_document(self, document_id: int) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM documents WHERE id=? AND project_id=?", (document_id,current_project_id())).rowcount > 0

    def create_run(self, interface: str, prompt: str, workflow_id: int | None = None):
        with self.connect() as db:
            cursor = db.execute("INSERT INTO runs(interface,workflow_id,prompt,project_id) VALUES (?,?,?,?)", (interface, workflow_id, prompt,current_project_id()))
            return dict(db.execute("SELECT * FROM runs WHERE id=?", (cursor.lastrowid,)).fetchone())

    def finish_run(self, run_id: int, status: str, result: dict | None = None, error: str = ""):
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,result_json=?,error=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=?", (status, json.dumps(result or {}), error, run_id,current_project_id()))
            row = db.execute("SELECT * FROM runs WHERE id=? AND project_id=?", (run_id,current_project_id())).fetchone()
            return dict(row) if row else None

    def cancel_run(self, run_id: int):
        with self.connect() as db:
            cursor = db.execute("UPDATE runs SET cancel_requested=1,status=CASE WHEN status='running' THEN 'cancelling' ELSE status END,updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=?", (run_id,current_project_id()))
            return cursor.rowcount > 0

    def run_cancelled(self, run_id: int) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT cancel_requested FROM runs WHERE id=? AND project_id=?", (run_id,current_project_id())).fetchone()
            return bool(row and row[0])

    def list_runs(self, limit: int = 50):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM runs WHERE project_id=? ORDER BY id DESC LIMIT ?", (current_project_id(),limit)).fetchall()
            return [{**dict(row), "result": json.loads(row["result_json"] or "{}")} for row in rows]

    def run_detail(self, run_id: int):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=? AND project_id=?", (run_id,current_project_id())).fetchone()
            return {**dict(row), "result": json.loads(row["result_json"] or "{}")} if row else None

    def delete_run(self, run_id: int) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM runs WHERE id=? AND project_id=?", (run_id,current_project_id())).rowcount > 0

    def get_settings(self):
        with self.connect() as db:
            return {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT key,value FROM app_settings")}

    def set_settings(self, values: dict):
        with self.connect() as db:
            for key, value in values.items():
                db.execute("""INSERT INTO app_settings(key,value) VALUES (?,?)
                  ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP""", (key, json.dumps(value)))
        return self.get_settings()
