"""Run the frozen protocol from the repository root."""

from pathlib import Path

from ttanco.experiment import load_research_config, run_research, save_research_report

config = load_research_config("configs/research_v1.json")
report = run_research(config, checkpoint_directory="artifacts/checkpoints")
output = Path("artifacts/research-report.json")
save_research_report(report, output)
print(output)
