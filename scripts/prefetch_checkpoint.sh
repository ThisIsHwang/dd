#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${ROOT}/checkpoints/openvla-oft-libero10}"
REPO="moojink/openvla-7b-oft-finetuned-libero-10"
REVISION="95220f9a3421a7ff12d4218e73d09ade830fa9a3"
python - "${REPO}" "${REVISION}" "${DEST}" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, revision, destination = sys.argv[1:]
path = Path(destination).expanduser().resolve()
path.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo, revision=revision, local_dir=str(path))
(path / "PROGRESSFLIP_REVISION.json").write_text(
    json.dumps({"repo": repo, "revision": revision}, indent=2) + "\n"
)
print(path)
PY
