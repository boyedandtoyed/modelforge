"""CLI for ModelForge fine-tuning operations."""
from __future__ import annotations
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

app = typer.Typer(name="modelforge", help="LLM fine-tuning toolkit")
console = Console()


@app.command()
def train(
    config_file: Path = typer.Option(None, "--config", "-c", help="JSON config file"),
    base_model: str = typer.Option("meta-llama/Llama-2-7b-hf", "--model", "-m"),
    dataset: str = typer.Option("tatsu-lab/alpaca", "--dataset", "-d"),
    epochs: int = typer.Option(3, "--epochs", "-e"),
    samples: int = typer.Option(1000, "--samples", "-n"),
    output: str = typer.Option("./outputs", "--output", "-o"),
):
    """Fine-tune an LLM with QLoRA."""
    from .config import TrainingConfig
    from .data import load_dataset
    from .trainer import train as run_train
    from .tracker import ExperimentTracker

    if config_file and config_file.exists():
        cfg = TrainingConfig(**json.loads(config_file.read_text()))
    else:
        cfg = TrainingConfig(
            base_model=base_model,
            dataset=dataset,
            num_epochs=epochs,
            max_samples=samples,
            output_dir=output,
        )

    console.print(f"\n[bold blue]ModelForge Training[/bold blue]")
    console.print(f"  Model:   {cfg.base_model}")
    console.print(f"  Dataset: {cfg.dataset}")
    console.print(f"  Epochs:  {cfg.num_epochs}")
    console.print(f"  LoRA r:  {cfg.lora.r}\n")

    console.print("[yellow]Loading dataset...[/yellow]")
    data = load_dataset(cfg.dataset, cfg.dataset_split, cfg.max_samples)
    console.print(f"[green]Loaded {len(data)} samples[/green]\n")

    tracker = ExperimentTracker(cfg.experiment_name)
    last_logged = [0]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = None

        def progress_cb(metrics, total_steps):
            nonlocal task
            if task is None:
                task = progress.add_task("Training...", total=total_steps)
            progress.update(task, advance=1, description=f"loss={metrics.loss:.4f} lr={metrics.learning_rate:.2e}")

        run = run_train(cfg, data, progress_cb)

    log_path = tracker.log_run(run)
    duration = (run.end_time - run.start_time) if run.end_time else 0

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Best loss:  {run.best_loss:.4f}")
    console.print(f"  Duration:   {duration:.1f}s")
    console.print(f"  Steps:      {len(run.metrics)}")
    console.print(f"  Log saved:  {log_path}")


@app.command()
def runs():
    """List all experiment runs."""
    from .tracker import ExperimentTracker
    tracker = ExperimentTracker("modelforge")
    all_runs = tracker.load_runs()
    if not all_runs:
        console.print("[yellow]No runs found.[/yellow]")
        return
    table = Table(title="Experiment Runs")
    table.add_column("Run ID")
    table.add_column("Model")
    table.add_column("Best Loss")
    table.add_column("Steps")
    table.add_column("Duration")
    for r in all_runs:
        cfg = r.get("config", {})
        table.add_row(
            r.get("run_id", "—"),
            cfg.get("base_model", "—")[:40],
            f"{r.get('best_loss', 0):.4f}",
            str(len(r.get("metrics", []))),
            f"{r.get('duration_seconds', 0):.1f}s",
        )
    console.print(table)


@app.command()
def serve(
    adapter: str = typer.Option("./outputs/adapter", "--adapter", "-a"),
    port: int = typer.Option(8080, "--port", "-p"),
):
    """Serve a fine-tuned model (vLLM in production)."""
    import uvicorn
    from .serve import app as serve_app
    console.print(f"[blue]Starting ModelForge server on port {port}[/blue]")
    console.print(f"[yellow]Demo mode — load adapter {adapter} with vLLM for real inference[/yellow]")
    uvicorn.run(serve_app, host="0.0.0.0", port=port)
