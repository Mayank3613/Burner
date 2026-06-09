import os
import json
from datetime import datetime, timezone
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from burner import run_burner_for_domain, BehaviorEntry, SufficiencyConfig, DivergenceType

app = typer.Typer(help="Burner — Behavioral-Narrative Divergence Scoring Engine CLI")
console = Console()

@app.command()
def run(
    behavior: str = typer.Option(..., "--behavior", "-b", help="Path to behavior log JSON file"),
    self_talk: str = typer.Option(..., "--self-talk", "-s", help="Path to self-talk JSON file"),
    domains: list[str] = typer.Option(..., "--domains", "-d", help="Domains to analyze"),
    output: str = typer.Option(..., "--output", "-o", help="Directory or file path to write results"),
    ref_date: str = typer.Option(
        None,
        "--ref-date",
        help="Reference ISO date for recency calculations (ISO 8601). Defaults to current UTC time."
    )
):
    """
    Runs the Burner engine on behavior logs and self-talk inputs,
    scores each domain, types the divergence, and outputs Pydantic-validated results.
    """
    console.print(Panel.fit("[bold orange1]BURNER[/bold orange1] — Behavioral-Narrative Divergence Scoring Engine", border_style="orange1"))
    
    # 1. Load inputs
    try:
        with open(behavior, "r") as f:
            behavior_data = json.load(f)
        with open(self_talk, "r") as f:
            self_talk_data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading input files:[/bold red] {e}", err=True)
        raise typer.Exit(code=1)
        
    # Convert behavior data to BehaviorEntry objects
    behavior_entries = []
    for item in behavior_data:
        try:
            behavior_entries.append(BehaviorEntry(**item))
        except Exception as e:
            console.print(f"[bold yellow]Skipping invalid behavior entry:[/bold yellow] {item}. Error: {e}", err=True)
            
    # Setup configuration
    ref_datetime = (
        datetime.fromisoformat(ref_date.replace("Z", "+00:00"))
        if ref_date
        else datetime.now(timezone.utc)
    )
    cfg = SufficiencyConfig(reference_date=ref_datetime)
    
    # 2. Run pipeline for each domain
    results = []
    for domain in domains:
        result = run_burner_for_domain(self_talk_data, behavior_entries, domain, cfg)
        results.append(result)
        
    # 3. Display results using a rich Table
    table = Table(title="Divergence Analysis Results", header_style="bold magenta")
    table.add_column("Domain", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Type", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Evidence", justify="right")
    table.add_column("Days", justify="right")
    table.add_column("Status", justify="left")
    
    for r in results:
        score_str = "—" if r.abstained else f"{r.divergence_score:+.2f}"
        
        if r.abstained:
            type_str = "[grey50]Insufficient Evidence[/grey50]"
            status_str = f"[yellow]Abstained · {r.abstention_reason}[/yellow]"
            conf_str = "—"
        else:
            status_str = "[green]Typed[/green]"
            conf_str = f"{r.confidence:.2f}"
            
            if r.divergence_type == DivergenceType.OVERSTATEMENT:
                type_str = "[bold orange1]Overstatement[/bold orange1]"
            elif r.divergence_type == DivergenceType.UNDERSTATEMENT:
                type_str = "[bold purple]Understatement[/bold purple]"
            elif r.divergence_type == DivergenceType.ASPIRATION_GAP:
                type_str = "[bold gold1]Aspiration Gap[/bold gold1]"
            elif r.divergence_type == DivergenceType.BLIND_SPOT:
                type_str = "[bold green]Blind Spot[/bold green]"
            else:
                type_str = "[bold blue]Aligned[/bold blue]"
                
        table.add_row(
            r.domain.capitalize(),
            score_str,
            type_str,
            conf_str,
            str(r.evidence_count),
            str(r.observation_days),
            status_str,
        )
        
    console.print(table)
    
    # 4. Save output
    os.makedirs(output, exist_ok=True)
    for r in results:
        # Determine filename
        if r.abstained:
            filename = f"{r.domain}_abstention_example.json"
        else:
            filename = f"{r.domain}_example.json"
            
        file_path = os.path.join(output, filename)
        try:
            with open(file_path, "w") as f:
                json.dump(r.model_dump(), f, indent=2)
            console.print(f"Result for domain '[bold cyan]{r.domain}[/bold cyan]' written to {file_path}")
        except Exception as e:
            console.print(f"[bold red]Failed to write result file for {r.domain}:[/bold red] {e}", err=True)

if __name__ == "__main__":
    app()
