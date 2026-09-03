"""Result I/O package for sharded cluster runs."""

from src.results_io.manifest import (
    SPLIT_PROTOCOL_VERSION,
    build_result_payload,
    code_version,
    config_hash,
    environment_fingerprint,
    environment_fingerprint_hash,
    git_commit_and_dirty,
)
from src.results_io.shards import (
    list_shards,
    make_run_id,
    read_shard,
    shard_path,
    shard_root,
    shard_success_exists,
    write_shard_atomic,
)

__all__ = [
    "SPLIT_PROTOCOL_VERSION",
    "build_result_payload",
    "code_version",
    "config_hash",
    "environment_fingerprint",
    "environment_fingerprint_hash",
    "git_commit_and_dirty",
    "list_shards",
    "make_run_id",
    "read_shard",
    "shard_path",
    "shard_root",
    "shard_success_exists",
    "write_shard_atomic",
]
