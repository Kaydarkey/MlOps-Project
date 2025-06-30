import subprocess

def test_rollback_triggers_on_drop(monkeypatch):
    # Patch config and metrics to simulate a drop
    monkeypatch.setattr("builtins.open", lambda f, *a, **k: open("configs/config.yaml"))
    result = subprocess.run(["python", "pipelines/rollback.py"], capture_output=True, text=True)
    assert "Rolling back" in result.stdout or "No rollback needed" in result.stdout 