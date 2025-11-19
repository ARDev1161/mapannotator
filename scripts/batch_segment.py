#!/usr/bin/env python3
"""
Batch runner for mapannotator CLI.

Usage:
    python scripts/batch_segment.py \
        --binary build/mapannotator \
        --maps-dir test_maps \
        --output-dir batch_results

The script scans the maps directory for *.pgm files, optionally picks the
matching *.yaml metadata, runs the CLI for each map and stores artefacts
(stdout/stderr, extracted PDDL, graph files) plus a YAML summary per map.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch test harness for the mapannotator CLI tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build") / "mapannotator",
        help="Path to the compiled mapannotator executable",
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("test_maps"),
        help="Directory containing *.pgm maps (optional *.yaml metadata)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_results"),
        help="Directory where per-map results will be written",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config passed as the second CLI argument",
    )
    parser.add_argument(
        "--clean-artifacts",
        action="store_true",
        help="Remove global artefacts (graph.dot, graph_preview.png) before every run",
    )
    return parser.parse_args()


def collect_maps(maps_dir: Path) -> List[Path]:
    if not maps_dir.is_dir():
        raise FileNotFoundError(f"Maps directory not found: {maps_dir}")
    pgm_files = sorted(maps_dir.glob("*.pgm"))
    if not pgm_files:
        raise RuntimeError(f"No *.pgm files found in {maps_dir}")
    return pgm_files


def run_cli(
    binary: Path,
    map_file: Path,
    meta_file: Optional[Path],
    config_file: Optional[Path],
    workdir: Path,
) -> subprocess.CompletedProcess:
    cmd = [str(binary), str(map_file)]
    if meta_file is not None and meta_file.is_file():
        cmd.append(str(meta_file))
    elif config_file is not None:
        cmd.append(str(config_file))
    # When both meta and config are provided, meta takes precedence as the CLI
    # expects the second argument to be the map yaml. Users can bake additional
    # config into default.yml if needed.

    result = subprocess.run(
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    return result


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def extract_pddl(stderr: str) -> str:
    start = stderr.find("(define ")
    if start == -1:
        return ""
    return stderr[start:]


def copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    if src.exists():
        shutil.copy2(src, dst)
        return dst
    return None


def clean_global_artifacts(workdir: Path, files: List[str]) -> None:
    for name in files:
        path = workdir / name
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)


def process_map(
    binary: Path,
    map_file: Path,
    config_file: Optional[Path],
    output_root: Path,
    workdir: Path,
    clean_artifacts_flag: bool,
) -> Dict[str, Optional[str]]:
    name = map_file.stem
    meta_file = map_file.with_suffix(".yaml")
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    global_artifacts = ["graph.dot", "graph_preview.png", "graph_preview.jpg"]
    if clean_artifacts_flag:
        clean_global_artifacts(workdir, global_artifacts)

    meta = meta_file if meta_file.exists() else None
    result = run_cli(binary, map_file, meta, config_file, workdir)

    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)

    pddl = extract_pddl(result.stderr)
    pddl_path = None
    if pddl:
        pddl_path = output_dir / f"{name}.pddl"
        write_text(pddl_path, pddl)

    copied_artifacts: Dict[str, Optional[str]] = {}
    for artifact in global_artifacts:
        src = workdir / artifact
        if src.exists():
            dst = output_dir / artifact
            copy_if_exists(src, dst)
            copied_artifacts[artifact] = str(dst)
        else:
            copied_artifacts[artifact] = None

    summary = {
        "map": str(map_file.resolve()),
        "metadata": str(meta_file.resolve()) if meta_file.exists() else None,
        "config": str(config_file.resolve()) if config_file else None,
        "exit_code": result.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "pddl": str(pddl_path) if pddl_path else None,
        "artifacts": copied_artifacts,
    }

    yaml_path = output_dir / f"{name}_summary.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(summary, fh, allow_unicode=True, sort_keys=False)

    return summary


def main() -> int:
    args = parse_args()
    binary = args.binary
    maps_dir = args.maps_dir
    output_dir = args.output_dir
    config_file = args.config

    if not binary.is_file():
        print(f"Binary not found: {binary}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    pgm_files = collect_maps(maps_dir)

    summaries: List[Dict[str, Optional[str]]] = []
    for pgm in pgm_files:
        print(f"[INFO] Processing {pgm.name} ...")
        summary = process_map(
            binary=binary,
            map_file=pgm,
            config_file=config_file,
            output_root=output_dir,
            workdir=binary.parent.resolve(),
            clean_artifacts_flag=args.clean_artifacts,
        )
        summaries.append(summary)
        if summary["exit_code"] != 0:
            print(textwrap.indent(Path(summary["stderr"]).read_text(), prefix="  "), file=sys.stderr)

    batch_summary_path = output_dir / "batch_summary.yaml"
    with batch_summary_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(summaries, fh, allow_unicode=True, sort_keys=False)

    print(f"[INFO] Processed {len(summaries)} map(s). Results stored in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
