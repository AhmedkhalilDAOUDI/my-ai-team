import json
import re
import subprocess
import difflib
from pathlib import Path, PurePosixPath

from .orchestrator import ThesisTeam


class BuilderError(RuntimeError):
    pass


class BuilderWorkspace:
    ALLOWED_TESTS = {
        "python": [".venv/bin/python", "-m", "pytest", "-q"],
        "pytest": [".venv/bin/pytest", "-q"],
        "npm": ["npm", "test", "--", "--runInBand"],
        "none": [],
    }

    def __init__(self, store, root: Path):
        self.store = store
        self.root = root.resolve()
        self.worktrees = self.root.parent / f".{self.root.name}-builder-worktrees"
        self.worktrees.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self):
        with self.store.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS builder_sessions (
              id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, task TEXT NOT NULL,
              branch TEXT NOT NULL, worktree TEXT NOT NULL, test_command TEXT NOT NULL DEFAULT 'python',
              status TEXT NOT NULL DEFAULT 'queued', plan TEXT NOT NULL DEFAULT '', review TEXT NOT NULL DEFAULT '',
              test_output TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS builder_events (
              id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES builder_sessions(id) ON DELETE CASCADE,
              stage TEXT NOT NULL, actor TEXT NOT NULL, message TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """)

    def _git(self, *args: str, cwd: Path | None = None, timeout: int = 120):
        result = subprocess.run(["git", *args], cwd=cwd or self.root, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise BuilderError((result.stderr or result.stdout).strip())
        return result.stdout.strip()

    def create(self, project_id: int, task: str, test_command: str):
        if test_command not in self.ALLOWED_TESTS: raise BuilderError("Unsupported test command")
        self._git("rev-parse", "--is-inside-work-tree")
        with self.store.connect() as db:
            cursor = db.execute(
                "INSERT INTO builder_sessions(project_id,task,branch,worktree,test_command) VALUES (?,?,?,?,?)",
                (project_id, task, "pending", "pending", test_command),
            )
            session_id = cursor.lastrowid
            branch = f"builder/session-{session_id}"
            worktree = self.worktrees / f"session-{session_id}"
            db.execute("UPDATE builder_sessions SET branch=?,worktree=? WHERE id=?",(branch,str(worktree),session_id))
        try:
            self._git("worktree", "add", "-b", branch, str(worktree), "HEAD")
        except Exception:
            with self.store.connect() as db: db.execute("DELETE FROM builder_sessions WHERE id=?",(session_id,))
            raise
        self.event(session_id,"created","System",f"Created isolated branch {branch}.")
        return self.detail(session_id, project_id)

    def event(self, session_id: int, stage: str, actor: str, message: str):
        with self.store.connect() as db:
            db.execute("INSERT INTO builder_events(session_id,stage,actor,message) VALUES (?,?,?,?)",(session_id,stage,actor,message[:8000]))

    def update(self, session_id: int, **values):
        allowed={"status","plan","review","test_output","error"}; values={k:v for k,v in values.items() if k in allowed}
        with self.store.connect() as db:
            db.execute(f"UPDATE builder_sessions SET {','.join(f'{k}=?' for k in values)},updated_at=CURRENT_TIMESTAMP WHERE id=?",(*values.values(),session_id))

    def list(self, project_id: int):
        with self.store.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM builder_sessions WHERE project_id=? ORDER BY id DESC",(project_id,))]

    def detail(self, session_id: int, project_id: int):
        with self.store.connect() as db:
            row=db.execute("SELECT * FROM builder_sessions WHERE id=? AND project_id=?",(session_id,project_id)).fetchone()
            if not row:return None
            events=[dict(item) for item in db.execute("SELECT * FROM builder_events WHERE session_id=? ORDER BY id",(session_id,))]
        session=dict(row); session["events"]=events
        path=Path(session["worktree"])
        session["diff"]=self.diff(path) if path.exists() else ""
        session["files"]=[line for line in self._git("status","--short",cwd=path).splitlines() if line] if path.exists() else []
        return session

    def diff(self, worktree: Path):
        patches=[self._git("diff","--no-ext-diff","--",cwd=worktree)]
        untracked=self._git("ls-files","--others","--exclude-standard",cwd=worktree).splitlines()
        for relative in untracked:
            target=self._safe_path(worktree,relative)
            try:lines=target.read_text(errors="replace").splitlines(keepends=True)
            except OSError:continue
            patches.append("".join(difflib.unified_diff([],lines,fromfile="/dev/null",tofile=f"b/{relative}")))
        return "\n".join(patch for patch in patches if patch)

    def _context(self, worktree: Path, limit: int = 70_000):
        parts=[]; used=0
        tracked=self._git("ls-files",cwd=worktree).splitlines()
        for relative in tracked:
            if relative.startswith((".venv/","data/")) or Path(relative).suffix.lower() in {".png",".jpg",".jpeg",".gif",".pdf",".db"}:continue
            path=worktree/relative
            try:text=path.read_text(errors="replace")
            except OSError:continue
            block=f"\n--- FILE: {relative} ---\n{text}"
            if used+len(block)>limit:break
            parts.append(block);used+=len(block)
        return "".join(parts)

    @staticmethod
    def _parse_changes(text: str):
        candidate=text.strip()
        fenced=re.search(r"```(?:json)?\s*(\{.*\})\s*```",candidate,re.S)
        if fenced:candidate=fenced.group(1)
        try:data=json.loads(candidate)
        except json.JSONDecodeError as exc:raise BuilderError("Builder returned invalid JSON changes") from exc
        changes=data.get("changes",[])
        if not isinstance(changes,list):raise BuilderError("Builder changes must be a list")
        return data.get("summary", ""), changes

    @staticmethod
    def _safe_path(root: Path, relative: str):
        pure=PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:raise BuilderError(f"Unsafe file path: {relative}")
        target=(root/Path(*pure.parts)).resolve()
        if root.resolve() not in target.parents:raise BuilderError(f"Unsafe file path: {relative}")
        return target

    def apply_changes(self, worktree: Path, changes: list[dict]):
        if len(changes)>30:raise BuilderError("A builder response may change at most 30 files")
        for change in changes:
            relative=str(change.get("path", "")); action=change.get("action","write")
            target=self._safe_path(worktree,relative)
            if action=="delete":
                if target.exists() and target.is_file():target.unlink()
            elif action=="write":
                content=change.get("content")
                if not isinstance(content,str):raise BuilderError(f"Missing content for {relative}")
                target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
            else:raise BuilderError(f"Unsupported action for {relative}")

    async def execute(self, session: dict, settings):
        session_id=session["id"];worktree=Path(session["worktree"]);team=ThesisTeam(settings)
        try:
            self.update(session_id,status="planning");self.event(session_id,"planning","Supervisor","Analyzing the repository and task.")
            context=self._context(worktree)
            supervisor_prompt=f"""TASK:\n{session['task']}\n\nREPOSITORY:\n{context}\n\nReturn only JSON: {{\"summary\":\"short plan\",\"changes\":[{{\"path\":\"relative/path\",\"action\":\"write|delete\",\"content\":\"complete file content for write\"}}]}}. Make the smallest complete implementation. Never change secrets, .env, .git, data, or dependency lockfiles unless essential."""
            initial=await team._safe_ask(team.providers["openai"],supervisor_prompt,"You are the implementation supervisor. Produce executable repository changes as strict JSON, not prose.")
            if initial.status!="ok":raise BuilderError(initial.error)
            plan,changes=self._parse_changes(initial.text);self.apply_changes(worktree,changes);self.update(session_id,plan=plan,status="reviewing")
            self.event(session_id,"implementation","Supervisor",plan or f"Applied {len(changes)} file changes.")
            diff=self.diff(worktree)
            critic_prompt=f"""TASK:\n{session['task']}\n\nCURRENT DIFF:\n{diff[:60000]}\n\nReturn only JSON in the same schema. Include only files that must be corrected, with their complete corrected content. If no correction is necessary, return {{\"summary\":\"Approved without changes\",\"changes\":[]}}."""
            review=await team._safe_ask(team.providers["deepseek"],critic_prompt,"You are the code critic and corrective builder. Review the diff, then return strict JSON containing only necessary corrective edits.")
            if review.status=="ok":
                review_summary,corrections=self._parse_changes(review.text);self.apply_changes(worktree,corrections);self.update(session_id,review=review_summary)
                self.event(session_id,"review","DeepSeek Critic",review_summary or f"Applied {len(corrections)} corrections.")
            else:self.event(session_id,"review","DeepSeek Critic",f"Review unavailable: {review.error}")
            self.update(session_id,status="testing");command=self.ALLOWED_TESTS[session["test_command"]]
            if command:
                executable=command[0]
                if executable.startswith(".venv/") and not (worktree/executable).exists():command=[str(self.root/executable),*command[1:]]
                result=subprocess.run(command,cwd=worktree,capture_output=True,text=True,timeout=300)
                output=(result.stdout+"\n"+result.stderr).strip()[-30000:]
                status="ready" if result.returncode==0 else "test_failed"
            else:output="Tests skipped by user selection.";status="ready"
            self.update(session_id,status=status,test_output=output);self.event(session_id,"testing","Test Runner",output or "Tests passed.")
        except Exception as exc:
            self.update(session_id,status="failed",error=str(exc));self.event(session_id,"failed","System",str(exc))

    def merge(self, session_id: int, project_id: int):
        session=self.detail(session_id,project_id)
        if not session or session["status"] not in {"ready","test_failed"}:raise BuilderError("Session is not ready to merge")
        worktree=Path(session["worktree"])
        if self._git("status","--porcelain",cwd=self.root):raise BuilderError("Main workspace has uncommitted changes; commit or stash them before merging")
        self._git("add","-A",cwd=worktree);self._git("commit","-m",f"Builder session {session_id}: {session['task'][:60]}",cwd=worktree)
        self._git("merge","--no-ff",session["branch"],"-m",f"Merge builder session {session_id}",cwd=self.root)
        self.update(session_id,status="merged");self.event(session_id,"merged","User","Approved and merged into the current branch.")
        return self.detail(session_id,project_id)
