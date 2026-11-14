"""Private, synchronous checkpoint commits; credentials are read by the HF SDK."""
import re
from pathlib import Path


def safe_path(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*", value):
        raise ValueError("Use a simple relative experiment/run ID")
    if any(part in (".", "..") for part in value.split("/")):
        raise ValueError("Invalid relative path")
    return value


class PrivateHub:
    def __init__(self, repo_id, run_id=None):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.repo_id = repo_id
        self.run_id = safe_path(run_id) if run_id else None
        self.api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")
        self.ensure_private()

    def ensure_private(self):
        if self.api.repo_info(self.repo_id, repo_type="model").private is not True:
            raise ValueError("Checkpoint repository must be private")

    def upload_run(self, folder):
        if self.run_id is None:
            raise ValueError("Set run_id before uploading checkpoints")
        self.ensure_private()
        self.api.upload_folder(repo_id=self.repo_id, folder_path=str(folder),
            path_in_repo="runs/" + self.run_id,
            allow_patterns=["latest.pt", "config.json", "metrics.json", "environment.json",
                            "external.json", "training.png"],
            commit_message="Text-JEPA v5 checkpoint: " + self.run_id)

    def download_run(self, folder):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError
        self.ensure_private()
        prefix = "runs/" + self.run_id
        if not self.api.file_exists(self.repo_id, prefix + "/latest.pt"):
            return False
        # Download to the same layout, then copy only this run's files.
        import shutil
        try:
            snapshot = snapshot_download(self.repo_id, allow_patterns=[prefix + "/*"])
        except (EntryNotFoundError, RemoteEntryNotFoundError):
            raise RuntimeError("Remote checkpoint changed during download; retry") from None
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for path in (Path(snapshot) / prefix).iterdir():
            if path.is_file():
                shutil.copy2(path, folder / path.name)
        return True

    def upload_corpus(self, folder, preparation_id):
        from .corpus import PreparedCorpus
        data = PreparedCorpus(folder)  # Verify all hashes before publishing.
        prefix = "prepared/" + safe_path(preparation_id)
        self.ensure_private()
        if self.api.file_exists(self.repo_id, prefix + "/manifest.json"):
            from huggingface_hub import hf_hub_download

            from .corpus import sha_file
            previous = hf_hub_download(self.repo_id, prefix + "/manifest.json")
            if sha_file(previous) != data.fingerprint:
                raise ValueError("Preparation ID already points to a different corpus")
            return
        self.api.upload_folder(repo_id=self.repo_id, folder_path=str(folder), path_in_repo=prefix,
            allow_patterns=["manifest.json", "sources.json", "*.bin", "*_metadata.jsonl", "tokenizer/*"],
            commit_message="Prepared Text-JEPA v5 corpus: " + preparation_id)

    def download_corpus(self, folder, preparation_id):
        import shutil

        from huggingface_hub import snapshot_download

        from .corpus import PreparedCorpus
        prefix = "prepared/" + safe_path(preparation_id)
        self.ensure_private()
        snapshot = snapshot_download(self.repo_id, allow_patterns=[prefix + "/**"])
        source = Path(snapshot) / prefix
        if not (source / "manifest.json").exists():
            raise FileNotFoundError("Prepared corpus is absent on the Hub")
        shutil.copytree(source, folder, dirs_exist_ok=True)
        return PreparedCorpus(folder)
