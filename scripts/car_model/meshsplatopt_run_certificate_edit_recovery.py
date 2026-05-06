#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a certificate edit recovery command stub.")
    parser.add_argument("--materialized_plan", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--recovery_command", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    command = shlex.join(args.recovery_command) if args.recovery_command else ""
    (out / "certificate_edit_recovery_command.txt").write_text(command + "\n", encoding="utf-8")
    (out / "real_surgery_report.md").write_text(
        "# Certificate Edit Recovery Report\n\n"
        f"- materialized plan: `{args.materialized_plan}`\n"
        f"- command written: `{bool(command)}`\n"
        "- execution: `manual/outer-runner`\n",
        encoding="utf-8",
    )
    print({"output_dir": str(out), "command_written": bool(command)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

