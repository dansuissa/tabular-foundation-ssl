
from __future__ import annotations
from src.results_io.shards import make_run_id, write_shard_atomic, list_shards, read_shard, shard_success_exists

def test_atomic_shard_write_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("SSL_SHARD_ROOT", str(tmp_path / "shards"))
    rid = make_run_id("phoneme","logistic_regression",0,50,"cfg","v1","nogit:x")
    path = write_shard_atomic("wave", rid, {"status":"success","run_id":rid,"dataset":"phoneme"})
    assert path.exists()
    assert shard_success_exists("wave", rid)
    data = read_shard(path)
    assert data["status"]=="success"
    assert len(list_shards("wave"))==1
    path2 = write_shard_atomic("wave", rid, {"status":"success","run_id":rid,"note":"overwrite"})
    assert path2 == path
    assert read_shard(path2)["note"]=="overwrite"
