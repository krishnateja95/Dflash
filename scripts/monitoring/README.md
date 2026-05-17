# Monitoring

Trackio helper scripts.

- `view_trackio.sh`: inspect local logs, print latest run metadata/metrics, and
  launch Trackio dashboard.

## Usage

```bash
bash scripts/monitoring/view_trackio.sh
```

Optional project override:

```bash
bash scripts/monitoring/view_trackio.sh speculators
```

If `trackio` is in a different environment, set:

```bash
export VENV_PATH="/path/to/venv/bin/activate"
```
