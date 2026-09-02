from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

QUEUE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClaimedWork:
    work_id: str
    task_id: int
    priority: float
    attempts: int
    payload: dict[str, Any]


class QueueError(RuntimeError):
    """Raised when a persisted work queue is inconsistent with its frozen workload."""


def default_worker_id(prefix: str = "worker") -> str:
    gpu = os.environ.get("PF_PHYSICAL_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"))
    slot = os.environ.get("PF_GPU_SLOT", "0")
    return f"{prefix}-{socket.gethostname()}-{os.getpid()}-g{gpu}-s{slot}"


def _config_source_fingerprint(cfg: Mapping[str, Any]) -> str:
    clean = {
        key: value
        for key, value in cfg.items()
        if not str(key).startswith("_") and key != "compute"
    }
    if isinstance(clean.get("data"), dict):
        clean["data"] = dict(clean["data"])
        clean["data"].pop("output_root", None)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60, isolation_level=None, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=60000")
    try:
        connection.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


class SQLiteWorkQueue:
    """Process-safe one-node lease queue stored on node-local NVMe."""

    def __init__(self, path: str | os.PathLike[str], lease_seconds: int = 3600) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lease_seconds = max(60, int(lease_seconds))
        self.connection = _connect(self.path)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS queue_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS work_items(
                work_id TEXT PRIMARY KEY,
                task_id INTEGER NOT NULL,
                priority REAL NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','running','done','failed','retired')),
                owner TEXT,
                lease_until REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_work_status_priority
                ON work_items(status, priority DESC, work_id ASC);
            CREATE INDEX IF NOT EXISTS idx_work_task_status
                ON work_items(task_id, status, priority DESC, work_id ASC);
            """
        )

    def __enter__(self) -> "SQLiteWorkQueue":
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    def _meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM queue_meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO queue_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def initialize(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        queue_kind: str,
        source_fingerprint: str,
        done_work_ids: Iterable[str] = (),
        reset_failed: bool = False,
        reclaim_running: bool = False,
    ) -> dict[str, Any]:
        done = {str(value) for value in done_work_ids}
        incoming = {str(item["work_id"]) for item in items}
        now = time.time()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing_kind = self._meta("queue_kind")
            existing_source = self._meta("source_fingerprint")
            if existing_kind is not None and existing_kind != queue_kind:
                raise QueueError(f"Queue belongs to {existing_kind!r}, not {queue_kind!r}")
            if existing_source is not None and existing_source != source_fingerprint:
                raise QueueError("Frozen workload changed; use a new PROGRESSFLIP_OUTPUT_ROOT")
            self._set_meta("schema_version", str(QUEUE_SCHEMA_VERSION))
            self._set_meta("queue_kind", queue_kind)
            self._set_meta("source_fingerprint", source_fingerprint)
            if reclaim_running:
                self.connection.execute(
                    "UPDATE work_items SET status='pending',owner=NULL,lease_until=NULL,"
                    "last_error=COALESCE(last_error,'reclaimed at phase restart'),updated_at=? "
                    "WHERE status='running'",
                    (now,),
                )
            else:
                self.connection.execute(
                    "UPDATE work_items SET status='pending',owner=NULL,lease_until=NULL,updated_at=? "
                    "WHERE status='running' AND (lease_until IS NULL OR lease_until < ?)",
                    (now, now),
                )
            if reset_failed:
                self.connection.execute(
                    "UPDATE work_items SET status='pending',owner=NULL,lease_until=NULL,"
                    "last_error=NULL,updated_at=? WHERE status='failed'",
                    (now,),
                )
            for item in items:
                work_id = str(item["work_id"])
                status = "done" if work_id in done else "pending"
                self.connection.execute(
                    """
                    INSERT INTO work_items(work_id,task_id,priority,payload_json,status,updated_at)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(work_id) DO UPDATE SET
                      task_id=excluded.task_id, priority=excluded.priority,
                      payload_json=excluded.payload_json,
                      status=CASE WHEN excluded.status='done' THEN 'done'
                                  WHEN work_items.status='done' THEN 'done'
                                  ELSE work_items.status END,
                      updated_at=excluded.updated_at
                    """,
                    (
                        work_id,
                        int(item.get("task_id", -1)),
                        float(item.get("priority", 0)),
                        json.dumps(dict(item.get("payload", {})), sort_keys=True),
                        status,
                        now,
                    ),
                )
            if incoming:
                placeholders = ",".join("?" for _ in incoming)
                self.connection.execute(
                    f"UPDATE work_items SET status='retired',owner=NULL,lease_until=NULL,updated_at=? "
                    f"WHERE work_id NOT IN ({placeholders}) AND status!='done'",
                    (now, *sorted(incoming)),
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return self.summary()

    def claim(self, owner: str, preferred_task_id: int | None = None) -> ClaimedWork | None:
        now = time.time()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "UPDATE work_items SET status='pending',owner=NULL,lease_until=NULL,"
                "last_error=COALESCE(last_error,'lease expired'),updated_at=? "
                "WHERE status='running' AND lease_until < ?",
                (now, now),
            )
            row = None
            if preferred_task_id is not None:
                row = self.connection.execute(
                    "SELECT * FROM work_items WHERE status='pending' AND task_id=? "
                    "ORDER BY priority DESC,work_id LIMIT 1",
                    (int(preferred_task_id),),
                ).fetchone()
            if row is None:
                row = self.connection.execute(
                    "SELECT * FROM work_items WHERE status='pending' "
                    "ORDER BY priority DESC,work_id LIMIT 1"
                ).fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None
            work_id = str(row["work_id"])
            cursor = self.connection.execute(
                "UPDATE work_items SET status='running',owner=?,lease_until=?,"
                "attempts=attempts+1,last_error=NULL,updated_at=? "
                "WHERE work_id=? AND status='pending'",
                (owner, now + self.lease_seconds, now, work_id),
            )
            if cursor.rowcount != 1:
                self.connection.execute("ROLLBACK")
                return None
            claimed = self.connection.execute(
                "SELECT * FROM work_items WHERE work_id=?", (work_id,)
            ).fetchone()
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        assert claimed is not None
        return ClaimedWork(
            work_id=work_id,
            task_id=int(claimed["task_id"]),
            priority=float(claimed["priority"]),
            attempts=int(claimed["attempts"]),
            payload=json.loads(str(claimed["payload_json"])),
        )

    def heartbeat(self, work_id: str, owner: str) -> None:
        cursor = self.connection.execute(
            "UPDATE work_items SET lease_until=?,updated_at=? "
            "WHERE work_id=? AND status='running' AND owner=?",
            (time.time() + self.lease_seconds, time.time(), work_id, owner),
        )
        if cursor.rowcount != 1:
            raise QueueError(f"Lost lease for {work_id!r}")

    def _finish(self, work_id: str, owner: str, status: str, error: str | None = None) -> None:
        cursor = self.connection.execute(
            "UPDATE work_items SET status=?,owner=NULL,lease_until=NULL,last_error=?,updated_at=? "
            "WHERE work_id=? AND status='running' AND owner=?",
            (status, error, time.time(), work_id, owner),
        )
        if cursor.rowcount != 1:
            raise QueueError(f"Cannot mark unowned work item {work_id!r} as {status}")

    def complete(self, work_id: str, owner: str) -> None:
        self._finish(work_id, owner, "done")

    def fail(self, work_id: str, owner: str, error: str) -> None:
        self._finish(work_id, owner, "failed", str(error)[-8000:])

    def summary(self) -> dict[str, Any]:
        observed = {
            str(row["status"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT status,COUNT(*) AS count FROM work_items GROUP BY status"
            )
        }
        running = [dict(row) for row in self.connection.execute(
            "SELECT work_id,task_id,owner,lease_until,attempts FROM work_items "
            "WHERE status='running' ORDER BY work_id"
        )]
        failed = [dict(row) for row in self.connection.execute(
            "SELECT work_id,attempts,last_error FROM work_items "
            "WHERE status='failed' ORDER BY work_id"
        )]
        return {
            "path": str(self.path),
            "kind": self._meta("queue_kind"),
            "source_fingerprint": self._meta("source_fingerprint"),
            "counts": {
                status: observed.get(status, 0)
                for status in ("pending", "running", "done", "failed", "retired")
            },
            "running": running,
            "failed": failed,
        }


def _completed_ids(root: Path, id_key: str, require_valid: bool) -> set[str]:
    completed: set[str] = set()
    if not root.is_dir():
        return completed
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("completed") and row.get(id_key):
                if not require_valid or row.get("valid", True):
                    completed.add(str(row[id_key]))
    return completed


def initialize_pair_queue(
    cfg: Mapping[str, Any],
    reset_failed: bool = False,
    reclaim_running: bool = False,
) -> dict[str, Any]:
    from .manifest import read_manifest

    output = Path(str(cfg["data"]["output_root"]))
    manifest_path = output / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing frozen manifest: {manifest_path}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_manifest(manifest_path):
        grouped.setdefault(str(row["pair_id"]), []).append(row)
    completed = _completed_ids(output / "results", "job_id", require_valid=True)
    done_pairs = {
        pair_id
        for pair_id, rows in grouped.items()
        if rows and all(str(row["job_id"]) in completed for row in rows)
    }
    max_steps = int(cfg["environment"].get("max_steps", 520))
    items = []
    for pair_id, rows in grouped.items():
        rows = sorted(rows, key=lambda value: str(value["condition"]))
        pair_json = Path(str(rows[0]["pair_dir"])) / "pair.json"
        stable_index = 0
        if pair_json.is_file():
            try:
                stable_index = int(json.loads(pair_json.read_text()).get("stable_index", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        items.append(
            {
                "work_id": pair_id,
                "task_id": int(rows[0]["task_id"]),
                "priority": float(len(rows) * max_steps + stable_index),
                "payload": {"jobs": rows},
            }
        )
    with SQLiteWorkQueue(
        output / "queues" / "run.sqlite3",
        int(cfg.get("compute", {}).get("lease_seconds", 3600)),
    ) as queue:
        return queue.initialize(
            items,
            queue_kind="run",
            source_fingerprint=_sha256(manifest_path),
            done_work_ids=done_pairs,
            reset_failed=reset_failed,
            reclaim_running=reclaim_running,
        )


def initialize_collection_queue(
    cfg: Mapping[str, Any],
    reset_failed: bool = False,
    reclaim_running: bool = False,
) -> dict[str, Any]:
    output = Path(str(cfg["data"]["output_root"]))
    count = int(cfg["data"].get("candidate_initial_states", 50))
    items = [
        {
            "work_id": f"collect--{task['key']}--init{initial_id:03d}",
            "task_id": int(task["task_id"]),
            "priority": float(count - initial_id),
            "payload": {
                "task_key": str(task["key"]),
                "task_id": int(task["task_id"]),
                "initial_state_id": initial_id,
            },
        }
        for task in cfg["tasks"]
        for initial_id in range(count)
    ]
    done = _completed_ids(output / "collection_results", "work_id", require_valid=False)
    with SQLiteWorkQueue(
        output / "queues" / "collect.sqlite3",
        int(cfg.get("compute", {}).get("lease_seconds", 3600)),
    ) as queue:
        return queue.initialize(
            items,
            queue_kind="collect",
            source_fingerprint=_config_source_fingerprint(cfg),
            done_work_ids=done,
            reset_failed=reset_failed,
            reclaim_running=reclaim_running,
        )


def queue_summary(cfg: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind not in {"collect", "run"}:
        raise ValueError(f"Unknown queue kind: {kind!r}")
    output = Path(str(cfg["data"]["output_root"]))
    path = output / "queues" / f"{kind}.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(path)
    with SQLiteWorkQueue(
        path, int(cfg.get("compute", {}).get("lease_seconds", 3600))
    ) as queue:
        return queue.summary()
