import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()
_log_console: Console | None = None
_log_path: Path | None = None


def setup_logger(run_name: str) -> Path:
    global _log_console, _log_path

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = logs_dir / f"{run_name}_{timestamp}.log"
    _log_console = Console(file=_log_path.open("w", encoding="utf-8"), force_terminal=False)
    console.log(f"Writing log to {_log_path}")
    return _log_path


def _write_log(renderable) -> None:
    if _log_console is not None:
        _log_console.print(renderable)


def render_score(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def log_api_base(api_base: str) -> None:
    console.log(f"Using Prometheus API at {api_base}")
    _write_log(f"Using Prometheus API at {api_base}")


def log_sample_count(sample_count: int) -> None:
    console.log(f"Evaluating {sample_count} sampled SummEval rows")
    _write_log(f"Evaluating {sample_count} sampled SummEval rows")


def print_feedback_panel(title: str, text: str) -> None:
    panel = Panel(text, title=title, expand=False)
    console.print(panel)
    _write_log(panel)


def print_score_table(title: str, rows: list[tuple[str, str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Domain", style="cyan")
    table.add_column("Human", style="green")
    table.add_column("Prometheus", style="magenta")
    for domain, human, prometheus in rows:
        table.add_row(domain, human, prometheus)
    console.print(table)
    _write_log(table)


def print_sample_header(idx: int, total: int) -> None:
    console.rule(f"Sample {idx}/{total}")
    _write_log(f"===== Sample {idx}/{total} =====")


def print_sample_metadata(sample: dict) -> None:
    table = Table(title="Sample Metadata")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("id", sample.get("id", ""))
    table.add_row("system_id", sample.get("system_id", ""))
    table.add_row(
        "human_scores",
        json.dumps(sample.get("human_scores", {}), sort_keys=True),
    )
    console.print(table)
    _write_log(table)


def log_progress(message: str) -> None:
    console.log(message)
    _write_log(message)
