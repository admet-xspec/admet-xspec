#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover .gin config files, generate one Slurm script per config, and submit them."
        )
    )
    parser.add_argument(
        "config_dir", type=Path, help="Directory containing .gin files."
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories for .gin files.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("data/cache/slurm_jobs"),
        help="Directory where generated Slurm scripts will be stored.",
    )
    parser.add_argument(
        "--partition", default="dgx_regular", help="Slurm partition/queue name."
    )
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", type=int, default=4)
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument(
        "--job-prefix",
        default="training",
        help="Prefix for generated Slurm job names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts and print sbatch commands without submitting.",
    )
    return parser.parse_args()


def find_gin_files(config_dir: Path, recursive: bool) -> list[Path]:
    if not config_dir.exists() or not config_dir.is_dir():
        raise FileNotFoundError(
            f"Config directory does not exist or is not a directory: {config_dir}"
        )

    pattern = "**/*.gin" if recursive else "*.gin"
    gin_files = sorted(
        path.resolve() for path in config_dir.glob(pattern) if path.is_file()
    )
    if not gin_files:
        raise FileNotFoundError(f"No .gin files found in: {config_dir}")
    return gin_files


def sanitize_job_name(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    sanitized = "".join(ch if ch in allowed else "_" for ch in name)
    return sanitized[:120]


def render_slurm_script(
    *,
    repo_root: Path,
    config_path: Path,
    job_name: str,
    output_path: Path,
    partition: str,
    nodes: int,
    ntasks_per_node: int,
    cpus_per_task: int,
) -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={output_path}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --ntasks-per-node={ntasks_per_node}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            "",
            f'cd "{repo_root}"',
            f'uv run process.py -c "{config_path}"',
            "",
        ]
    )


def write_scripts(
    *,
    gin_files: Iterable[Path],
    jobs_dir: Path,
    repo_root: Path,
    partition: str,
    nodes: int,
    ntasks_per_node: int,
    cpus_per_task: int,
    job_prefix: str,
) -> list[Path]:
    # append the date to jobs_dir to avoid overwriting previous runs
    today_is = datetime.today().strftime("%Y-%m-%d")
    jobs_dir = jobs_dir / today_is
    jobs_dir.mkdir(parents=True, exist_ok=True)

    script_paths: list[Path] = []
    for cfg in gin_files:
        cfg_stem = cfg.stem
        job_name = sanitize_job_name(f"{job_prefix}_{cfg_stem}")
        output_path = jobs_dir / f"{job_name}.out"
        script_path = jobs_dir / f"{job_name}.sbatch"

        script_text = render_slurm_script(
            repo_root=repo_root,
            config_path=cfg,
            job_name=job_name,
            output_path=output_path,
            partition=partition,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            cpus_per_task=cpus_per_task,
        )
        script_path.write_text(script_text, encoding="utf-8")
        script_paths.append(script_path)

    return script_paths


def submit_scripts(script_paths: Iterable[Path], dry_run: bool) -> None:
    for script_path in script_paths:
        cmd = ["sbatch", str(script_path)]
        if dry_run:
            print("DRY RUN:", " ".join(cmd))
            continue
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout.strip())


def main() -> None:
    args = parse_args()

    recursive = not args.no_recursive
    gin_files = find_gin_files(args.config_dir, recursive=recursive)

    repo_root = Path(__file__).resolve().parents[1]

    scripts = write_scripts(
        gin_files=gin_files,
        jobs_dir=args.jobs_dir.resolve(),
        repo_root=repo_root,
        partition=args.partition,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        cpus_per_task=args.cpus_per_task,
        job_prefix=args.job_prefix,
    )

    print(f"Found {len(gin_files)} config file(s).")
    print(f"Generated {len(scripts)} Slurm script(s) in: {args.jobs_dir.resolve()}")
    submit_scripts(scripts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
