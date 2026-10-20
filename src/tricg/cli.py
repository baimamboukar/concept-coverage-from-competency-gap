"""Command-line interface for coverage profiles and benchmark matrices."""

import argparse
import json
from pathlib import Path

from tricg.coverage import build_matrix, build_profile
from tricg.importers import stage1_activations


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="tricg", description="Build concept coverage profiles")
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile", help="Aggregate token-mean activations")
    profile.add_argument("input", type=Path, help="Benchmark activations and extraction metadata")
    profile.add_argument("--stage1-csv", type=Path, help="Import the authors' activation CSV")
    matrix = commands.add_parser("matrix", help="Stack profiles in a shared concept space")
    matrix.add_argument("profiles", nargs="+", type=Path)
    for command in (profile, matrix):
        command.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "profile":
            data = _read(args.input)
            activations = (
                stage1_activations(args.stage1_csv) if args.stage1_csv else data["activations"]
            )
            result = build_profile(
                data["benchmark"], activations, data["feature_count"], data["instrument"]
            )
            if "source" in data:
                result["source"] = data["source"]
        else:
            result = build_matrix([_read(path) for path in args.profiles])
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(content, encoding="utf-8")
        else:
            print(content, end="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{parser.prog} error — {error}\n")


if __name__ == "__main__":
    main()
