import math
import re
import json
import httpx
import hashlib
import secrets
from collections import Counter
from datetime import datetime, timezone
from .storage import Store, current_project_id

TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]{2,}")
ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){0,3})\b")


def chunk_text(text: str, size: int = 1200, overlap: int = 180):
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current: chunks.append(current)
    return chunks or [text[:size]]


def terms(text: str): return [token.lower() for token in TOKEN_RE.findall(text)]


class PlatformStore:
    def __init__(self, store: Store):
        self.store = store
        self.initialize()

    def initialize(self):
        with self.store.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS document_chunks (
              id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
              project_id INTEGER NOT NULL DEFAULT 1 REFERENCES projects(id) ON DELETE CASCADE,
              chunk_index INTEGER NOT NULL, content TEXT NOT NULL, token_count INTEGER NOT NULL,
              UNIQUE(document_id,chunk_index)
            );
            CREATE TABLE IF NOT EXISTS graph_entities (
              id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              name TEXT NOT NULL, normalized TEXT NOT NULL, mention_count INTEGER NOT NULL DEFAULT 1,
              UNIQUE(project_id,normalized)
            );
            CREATE TABLE IF NOT EXISTS graph_relationships (
              id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              source TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL, document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
              evidence TEXT NOT NULL, UNIQUE(project_id,source,relation,target,document_id)
            );
            CREATE TABLE IF NOT EXISTS prompt_versions (
              id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
              role TEXT NOT NULL, instructions TEXT NOT NULL, version INTEGER NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(agent_id,version)
            );
            CREATE TABLE IF NOT EXISTS evaluation_suites (
              id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS evaluation_cases (
              id INTEGER PRIMARY KEY, suite_id INTEGER NOT NULL REFERENCES evaluation_suites(id) ON DELETE CASCADE,
              question TEXT NOT NULL, expected TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS evaluation_results (
              id INTEGER PRIMARY KEY, suite_id INTEGER NOT NULL REFERENCES evaluation_suites(id) ON DELETE CASCADE,
              case_id INTEGER NOT NULL REFERENCES evaluation_cases(id) ON DELETE CASCADE,
              workflow_id INTEGER REFERENCES workflows(id) ON DELETE SET NULL, answer TEXT NOT NULL,
              score REAL NOT NULL, latency_ms INTEGER NOT NULL, cost_usd REAL NOT NULL DEFAULT 0,
              details TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS jobs (
              id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              kind TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
              result TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS provider_plugins (
              id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, adapter TEXT NOT NULL, base_url TEXT NOT NULL DEFAULT '',
              api_key_env TEXT NOT NULL DEFAULT '', enabled INTEGER NOT NULL DEFAULT 0, config TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, token_hash TEXT NOT NULL UNIQUE,
              is_admin INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS project_members (
              user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
              role TEXT NOT NULL DEFAULT 'member', PRIMARY KEY(user_id,project_id)
            );
            """)
            db.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)")
            for table in ("documents", "runs", "conversations", "agents", "usage_records"):
                columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
                if "project_id" not in columns: db.execute(f"ALTER TABLE {table} ADD COLUMN project_id INTEGER NOT NULL DEFAULT 1")
            db.execute("INSERT OR IGNORE INTO provider_plugins(name,adapter,enabled) VALUES ('OpenAI','builtin-openai',1)")
            db.execute("INSERT OR IGNORE INTO provider_plugins(name,adapter,enabled) VALUES ('DeepSeek','builtin-deepseek',1)")
            missing = [row[0] for row in db.execute("""SELECT d.id FROM documents d LEFT JOIN document_chunks c ON c.document_id=d.id
              WHERE c.id IS NULL""").fetchall()]
        for document_id in missing: self.index_document(document_id)
        with self.store.connect() as db:
            chunk_columns={row["name"] for row in db.execute("PRAGMA table_info(document_chunks)")}
            if "embedding" not in chunk_columns: db.execute("ALTER TABLE document_chunks ADD COLUMN embedding TEXT")
            if "embedding_model" not in chunk_columns: db.execute("ALTER TABLE document_chunks ADD COLUMN embedding_model TEXT")

    def index_document(self, document_id: int, project_id: int = 1):
        with self.store.connect() as db:
            document = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            if not document: return None
            db.execute("DELETE FROM document_chunks WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM graph_relationships WHERE document_id=?", (document_id,))
            chunks = chunk_text(document["content"])
            for index, content in enumerate(chunks):
                db.execute("INSERT INTO document_chunks(document_id,project_id,chunk_index,content,token_count) VALUES (?,?,?,?,?)", (document_id,project_id,index,content,len(terms(content))))
                entities = list(dict.fromkeys(ENTITY_RE.findall(content)))[:30]
                for entity in entities:
                    normalized = entity.lower()
                    db.execute("""INSERT INTO graph_entities(project_id,name,normalized,mention_count) VALUES (?,?,?,1)
                      ON CONFLICT(project_id,normalized) DO UPDATE SET mention_count=mention_count+1""", (project_id,entity,normalized))
                for source, target in zip(entities, entities[1:]):
                    db.execute("INSERT OR IGNORE INTO graph_relationships(project_id,source,relation,target,document_id,evidence) VALUES (?,?,?,?,?,?)", (project_id,source,"CO_OCCURS_WITH",target,document_id,content[:500]))
            return {"document_id": document_id, "chunks": len(chunks)}

    def retrieve(self, query: str, project_id: int = 1, document_ids: list[int] | None = None, limit: int = 6):
        with self.store.connect() as db:
            params = [project_id]; where = "c.project_id=?"
            if document_ids:
                where += f" AND c.document_id IN ({','.join('?' for _ in document_ids)})"; params.extend(document_ids)
            rows = db.execute(f"""SELECT c.*,d.filename FROM document_chunks c JOIN documents d ON d.id=c.document_id
              WHERE {where}""", params).fetchall()
        query_terms = Counter(terms(query)); total = max(1, len(rows)); document_frequency = Counter()
        row_terms = []
        for row in rows:
            counts = Counter(terms(row["content"])); row_terms.append(counts)
            for token in counts: document_frequency[token] += 1
        ranked = []
        for row, counts in zip(rows, row_terms):
            score = sum((1 + math.log(counts[token])) * math.log(1 + total/(1+document_frequency[token])) * weight for token,weight in query_terms.items() if counts[token])
            if score > 0: ranked.append({**dict(row), "score": round(score, 5), "citation": f"D{row['document_id']}C{row['chunk_index']}"})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]

    async def embed_document(self, document_id: int, api_key: str | None, model: str):
        if not api_key: return {"embedded":0,"mode":"keyword"}
        with self.store.connect() as db: rows=db.execute("SELECT id,content FROM document_chunks WHERE document_id=? ORDER BY chunk_index",(document_id,)).fetchall()
        if not rows:return {"embedded":0,"mode":"keyword"}
        async with httpx.AsyncClient(timeout=60) as client:
            response=await client.post("https://api.openai.com/v1/embeddings",headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},json={"model":model,"input":[row["content"] for row in rows]})
            response.raise_for_status();vectors=response.json().get("data") or []
        with self.store.connect() as db:
            for row,item in zip(rows,vectors):db.execute("UPDATE document_chunks SET embedding=?,embedding_model=? WHERE id=?",(json.dumps(item["embedding"]),model,row["id"]))
        return {"embedded":len(vectors),"mode":"hybrid","model":model}

    async def hybrid_retrieve(self, query: str, project_id: int, document_ids: list[int] | None, limit: int, api_key: str | None, model: str):
        lexical=self.retrieve(query,project_id,document_ids,max(limit*3,limit))
        if not api_key:return lexical[:limit]
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response=await client.post("https://api.openai.com/v1/embeddings",headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},json={"model":model,"input":query});response.raise_for_status();query_vector=response.json()["data"][0]["embedding"]
            with self.store.connect() as db:
                params=[project_id,model];where="project_id=? AND embedding_model=? AND embedding IS NOT NULL"
                if document_ids:where+=f" AND document_id IN ({','.join('?' for _ in document_ids)})";params.extend(document_ids)
                rows=db.execute(f"SELECT c.*, (SELECT filename FROM documents d WHERE d.id=c.document_id) filename FROM document_chunks c WHERE {where}",params).fetchall()
            semantic=[];qnorm=math.sqrt(sum(v*v for v in query_vector)) or 1
            for row in rows:
                vector=json.loads(row["embedding"]);norm=math.sqrt(sum(v*v for v in vector)) or 1;score=sum(a*b for a,b in zip(query_vector,vector))/(qnorm*norm)
                semantic.append({**dict(row),"score":score,"citation":f"D{row['document_id']}C{row['chunk_index']}"})
            combined={item["id"]:{**item,"score":0.35*item["score"]} for item in lexical}
            for item in semantic:
                if item["id"] in combined:combined[item["id"]]["score"]+=0.65*item["score"]
                else:combined[item["id"]]={**item,"score":0.65*item["score"]}
            return sorted(combined.values(),key=lambda item:item["score"],reverse=True)[:limit]
        except Exception:
            return lexical[:limit]

    def graph(self, project_id: int = 1, limit: int = 200):
        with self.store.connect() as db:
            entities = [dict(row) for row in db.execute("SELECT * FROM graph_entities WHERE project_id=? ORDER BY mention_count DESC LIMIT ?", (project_id,limit))]
            relationships = [dict(row) for row in db.execute("SELECT * FROM graph_relationships WHERE project_id=? ORDER BY id DESC LIMIT ?", (project_id,limit))]
            return {"entities": entities, "relationships": relationships}

    def graph_context(self, query: str, project_id: int = 1, limit: int = 8):
        query_tokens = set(terms(query))
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM graph_relationships WHERE project_id=? ORDER BY id DESC", (project_id,)).fetchall()
        matches = []
        for row in rows:
            names = set(terms(row["source"] + " " + row["target"]))
            if names & query_tokens:
                matches.append(f"{row['source']} --{row['relation']}--> {row['target']} [D{row['document_id']}]" )
            if len(matches) >= limit: break
        return matches

    def record_prompt_version(self, agent_id: int, role: str, instructions: str):
        with self.store.connect() as db:
            version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM prompt_versions WHERE agent_id=?", (agent_id,)).fetchone()[0]
            db.execute("INSERT INTO prompt_versions(agent_id,role,instructions,version) VALUES (?,?,?,?)", (agent_id,role,instructions,version))
            return version

    def prompt_versions(self, agent_id: int):
        with self.store.connect() as db: return [dict(row) for row in db.execute("""SELECT pv.* FROM prompt_versions pv JOIN agents a ON a.id=pv.agent_id
          WHERE pv.agent_id=? AND a.project_id=? ORDER BY pv.version DESC""", (agent_id,current_project_id()))]

    def plugins(self):
        with self.store.connect() as db: return [dict(row) for row in db.execute("SELECT * FROM provider_plugins ORDER BY id")]

    def plugin(self, name: str):
        with self.store.connect() as db:
            row=db.execute("SELECT * FROM provider_plugins WHERE name=?",(name,)).fetchone()
            return dict(row) if row else None

    def create_suite(self, project_id: int, name: str, description: str):
        with self.store.connect() as db:
            cursor=db.execute("INSERT INTO evaluation_suites(project_id,name,description) VALUES (?,?,?)",(project_id,name,description))
            suite_id = cursor.lastrowid
        return self.suite(suite_id)

    def add_case(self, suite_id: int, question: str, expected: str, tags: str):
        with self.store.connect() as db:
            cursor=db.execute("INSERT INTO evaluation_cases(suite_id,question,expected,tags) VALUES (?,?,?,?)",(suite_id,question,expected,tags))
            return dict(db.execute("SELECT * FROM evaluation_cases WHERE id=?",(cursor.lastrowid,)).fetchone())

    def suites(self, project_id: int = 1):
        with self.store.connect() as db:
            return [dict(row) for row in db.execute("""SELECT s.*,COUNT(c.id) case_count FROM evaluation_suites s
              LEFT JOIN evaluation_cases c ON c.suite_id=s.id WHERE s.project_id=? GROUP BY s.id ORDER BY s.id DESC""",(project_id,))]

    def suite(self, suite_id: int):
        with self.store.connect() as db:
            suite=db.execute("SELECT * FROM evaluation_suites WHERE id=? AND project_id=?",(suite_id,current_project_id())).fetchone()
            if not suite:return None
            cases=[dict(row) for row in db.execute("SELECT * FROM evaluation_cases WHERE suite_id=? ORDER BY id",(suite_id,))]
            results=[dict(row) for row in db.execute("SELECT * FROM evaluation_results WHERE suite_id=? ORDER BY id DESC",(suite_id,))]
            return {**dict(suite),"cases":cases,"results":results}

    def record_evaluation(self, suite_id: int, case_id: int, workflow_id: int, answer: str, score: float, latency_ms: int, cost_usd: float, details: dict):
        with self.store.connect() as db:
            cursor=db.execute("""INSERT INTO evaluation_results(suite_id,case_id,workflow_id,answer,score,latency_ms,cost_usd,details)
              VALUES (?,?,?,?,?,?,?,?)""",(suite_id,case_id,workflow_id,answer,score,latency_ms,cost_usd,json.dumps(details)))
            return dict(db.execute("SELECT * FROM evaluation_results WHERE id=?",(cursor.lastrowid,)).fetchone())

    def create_job(self, project_id: int, kind: str, payload: dict):
        with self.store.connect() as db:
            cursor=db.execute("INSERT INTO jobs(project_id,kind,payload) VALUES (?,?,?)",(project_id,kind,json.dumps(payload)))
            return dict(db.execute("SELECT * FROM jobs WHERE id=?",(cursor.lastrowid,)).fetchone())

    def jobs(self, limit: int = 100):
        with self.store.connect() as db:return [dict(row) for row in db.execute("SELECT * FROM jobs WHERE project_id=? ORDER BY id DESC LIMIT ?",(current_project_id(),limit))]

    def claim_job(self):
        with self.store.connect() as db:
            row=db.execute("SELECT * FROM jobs WHERE status IN ('pending','retrying') ORDER BY id LIMIT 1").fetchone()
            if not row:return None
            db.execute("UPDATE jobs SET status='running',attempts=attempts+1,updated_at=CURRENT_TIMESTAMP WHERE id=?",(row["id"],))
            return {**dict(row),"payload_data":json.loads(row["payload"])}

    def finish_job(self, job_id: int, status: str, result: dict | None = None, error: str = ""):
        with self.store.connect() as db:
            db.execute("UPDATE jobs SET status=?,result=?,error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",(status,json.dumps(result or {}),error,job_id))

    def cancel_job(self, job_id: int):
        with self.store.connect() as db:return db.execute("UPDATE jobs SET status='cancelled',updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=? AND status IN ('pending','retrying')",(job_id,current_project_id())).rowcount>0

    def save_plugin(self, name: str, adapter: str, base_url: str, api_key_env: str, enabled: bool, config: dict):
        with self.store.connect() as db:
            db.execute("""INSERT INTO provider_plugins(name,adapter,base_url,api_key_env,enabled,config) VALUES (?,?,?,?,?,?)
              ON CONFLICT(name) DO UPDATE SET adapter=excluded.adapter,base_url=excluded.base_url,api_key_env=excluded.api_key_env,enabled=excluded.enabled,config=excluded.config""",(name,adapter,base_url,api_key_env,int(enabled),json.dumps(config)))
        return next(item for item in self.plugins() if item["name"]==name)

    @staticmethod
    def token_hash(token: str): return hashlib.sha256(token.encode()).hexdigest()

    def create_user(self, username: str, project_name: str):
        token=secrets.token_urlsafe(32)
        with self.store.connect() as db:
            project_id=db.execute("INSERT INTO projects(name,description) VALUES (?,?)",(project_name,f"Private workspace for {username}")).lastrowid
            user_id=db.execute("INSERT INTO users(username,token_hash) VALUES (?,?)",(username,self.token_hash(token))).lastrowid
            db.execute("INSERT INTO project_members(user_id,project_id,role) VALUES (?,?,?)",(user_id,project_id,"owner"))
            agents = [
                ("Supervisor", "openai", "gpt-5.4", "Primary Supervisor", "Answer first with the strongest direct recommendation and concrete next steps."),
                ("Critic", "deepseek", "deepseek-reasoner", "Critic & Reviewer", "Review the supervisor's answer, identify flaws, and propose precise corrections."),
                ("Jury", "openai", "gpt-5.4", "Jury", "Only ask the most important unresolved questions. Do not answer them."),
                ("Completeness", "deepseek", "deepseek-reasoner", "Completeness Checker", "Identify important omissions and missing risks without repeating prior points."),
            ]
            agent_ids = []
            for name, provider, model, role, prompt in agents:
                agent_ids.append(db.execute(
                    "INSERT INTO agents(project_id,name,provider,model,role,instructions,max_sentences,can_read_peers,enabled) VALUES (?,?,?,?,?,?,?,?,?)",
                    (project_id,name,provider,model,role,prompt,5,1,1),
                ).lastrowid)
            workflow_id = db.execute(
                "INSERT INTO workflows(project_id,name,description) VALUES (?,?,?)",
                (project_id,"Direct Workspace","Supervisor, critic, jury, and completeness review"),
            ).lastrowid
            for position, agent_id in enumerate(agent_ids, start=1):
                db.execute("INSERT INTO workflow_steps(workflow_id,agent_id,position) VALUES (?,?,?)",(workflow_id,agent_id,position))
        return {"id":user_id,"username":username,"project_id":project_id,"access_token":token}

    def authenticate(self, token: str):
        with self.store.connect() as db:
            row=db.execute("""SELECT u.id,u.username,u.is_admin,pm.project_id,pm.role FROM users u
              JOIN project_members pm ON pm.user_id=u.id WHERE u.token_hash=? ORDER BY pm.project_id LIMIT 1""",(self.token_hash(token),)).fetchone()
            return dict(row) if row else None

    def users(self):
        with self.store.connect() as db:
            return [dict(row) for row in db.execute("""SELECT u.id,u.username,u.is_admin,pm.project_id,pm.role,u.created_at FROM users u
              JOIN project_members pm ON pm.user_id=u.id ORDER BY u.id""")]
