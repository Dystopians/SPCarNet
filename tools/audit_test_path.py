#!/usr/bin/env python
"""GEMS D4 purity audit for the single evaluation mouth (PROTOCOL.md sections 4 and 6).

Two checks; BOTH must pass (exit 0) for the audit to be green:

1. STATIC — AST-walks ``run_eval.py`` plus every module under ``tools/gems/``
   (core files), and every repo module they import, one level deep
   (transitive files). FAILS if any import statement in any walked file
   matches the purity blocklist regex

       (evidence_lumigraph|teacher|ecsr_|phasej|phase_j|selector|calibrator|
        arbitrat|scripts[./\\\\]car_model|car_model|lumigraph)

   ``scripts[./\\\\]car_model`` matches the dotted, slash and backslash forms;
   the standalone ``car_model`` token additionally catches bare module names
   and paths that mention the package without the ``scripts`` prefix.

   Additionally greps the same files for suspicious stub tokens (see
   ``WARN_TOKEN_RE``). These are WARN-level, except a
   ``raise NotImplementedError`` statement inside a CORE file, which FAILS
   (heuristic for "non-optional path": core eval files must not contain any;
   transitive repo utility modules only get a warning).

2. DYNAMIC — runs ``run_eval.py`` in a subprocess with a ``sitecustomize.py``
   hook (prepended via PYTHONPATH) that registers an atexit dump of
   ``{module_name: getattr(module, '__file__', None)}`` to a JSON file.
   FAILS if the eval process exits non-zero, if the module dump is missing,
   or if any loaded module violates purity by NAME or by FILE:
     - name: dotted module name matches the blocklist;
     - file: the module's ``__file__`` realpath is under
       ``<repo>/scripts/car_model/`` OR matches the blocklist — regardless of
       the module name. This catches bare-name imports after a
       ``sys.path.insert`` (e.g. ``import run_l1risk_fairnoop_scene`` with
       ``scripts/car_model`` on sys.path), which the name check alone misses.
   Documented false-positive exemption: the Python stdlib modules
   ``selectors`` and ``*.selector_events`` (asyncio) match the ``selector``
   token but are benign; only those exact segment names are stripped before
   matching (a repo module such as
   ``ecsr_run_facelocal_coupled_selector`` still fails).

   Best-effort read audit: when ``strace`` is available the subprocess is
   traced (``-f -e trace=open,openat,openat2``) and every successfully opened
   read path is checked. A read whose realpath is under
   ``<repo>/scripts/car_model/`` FAILS unconditionally; a read whose realpath
   matches the blocklist regex FAILS (paths under the Python installation and
   OS system roots are exempt so the stdlib's ``selectors.py`` does not trip
   it); reads outside the documented allowed roots (repo, python env, system
   dirs, home caches, checkpoint, scene data, out dir, temp) are reported as
   WARNINGS only. If strace is missing or cannot attach, this sub-check is
   skipped with a note and never fails the audit (per the protocol:
   best-effort).

Usage:
    python tools/audit_test_path.py --checkpoint <point_cloud_state_dict.pt> \
        --scene <toy_parking|garden|courtyard> --out <dir> [--fast]

``--fast`` additionally passes ``--skip-geometry --skip-downstream`` to
run_eval.py for a quick purity-only pass.

Exit codes: 0 = green, 1 = red. Reasons are printed and the full report is
written to ``<out>/audit_report.json``.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLOCKLIST_PATTERN = (
    r"(evidence_lumigraph|teacher|ecsr_|phasej|phase_j|selector|calibrator|"
    # scripts[./\]car_model matches the dotted (scripts.car_model), slash
    # (scripts/car_model) and backslash (scripts\car_model) forms; the
    # standalone car_model token also catches bare names/paths without the
    # scripts prefix (kept separate for documentation even though it
    # subsumes the prefixed form).
    r"arbitrat|scripts[./\\]car_model|car_model|lumigraph)"
)
BLOCKLIST_RE = re.compile(BLOCKLIST_PATTERN)

# Directory whose contents are NEVER allowed in the eval process, whatever the
# module is called: any loaded module __file__ or successfully-read path whose
# realpath lands here FAILS the audit.
SCRIPTS_CAR_MODEL_ROOT = os.path.realpath(
    os.path.join(REPO_ROOT, "scripts", "car_model"))

# Stub-token greplist: TO-DO / FIX-ME / mo-ck / place-holder / NotImplemented.
# Assembled from fragments so this file does not itself contain the literal
# tokens (the Stage-Two audit greps core paths for them).
WARN_TOKEN_RE = re.compile("|".join(
    ["TO" "DO", "FIX" "ME", "mo" "ck", "place" "holder", "Not" "Implemented"]))

# Python-stdlib segment names that legitimately match the 'selector' token.
# Only these exact segments are stripped before blocklist matching.
_STDLIB_FALSE_POSITIVE_SEGMENTS = {"selectors", "selector_events"}

# OS / interpreter roots exempt from the blocklist match on read paths
# (the stdlib ships e.g. asyncio/selector_events.py).
_SYSTEM_ROOTS = ("/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev",
                 "/run", "/opt")


def _strip_stdlib_segments(dotted_or_path: str, sep: str) -> str:
    parts = dotted_or_path.split(sep)
    kept = [p for p in parts
            if p.split(".py")[0] not in _STDLIB_FALSE_POSITIVE_SEGMENTS]
    return sep.join(kept)


def module_name_violates(name: str) -> bool:
    """Blocklist match on a dotted module name, exempting stdlib selectors."""
    return bool(BLOCKLIST_RE.search(_strip_stdlib_segments(name, ".")))


# --------------------------------------------------------------------------
# STATIC check
# --------------------------------------------------------------------------

def _import_candidates(tree: ast.AST, package: str):
    """Yield (candidate_module_string, lineno) for every import in *tree*.

    ``package`` is the dotted package of the file being walked (used to
    absolutize relative imports); '' for top-level scripts.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level and package:
                pkg_parts = package.split(".")
                base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                base = ".".join(base_parts)
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            if mod:
                yield mod, node.lineno
            for alias in node.names:
                if alias.name != "*":
                    yield (f"{mod}.{alias.name}" if mod else alias.name), node.lineno


def _package_of(path: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), REPO_ROOT)
    if rel.startswith(".."):
        return ""
    parts = rel.split(os.sep)[:-1]  # drop filename -> containing package
    return ".".join(parts)


def resolve_repo_module(modstr: str, repo_root: str = REPO_ROOT):
    """Map a dotted module string to a repo source file, if it is a repo module."""
    parts = modstr.split(".")
    for n in range(len(parts), 0, -1):
        base = os.path.join(repo_root, *parts[:n])
        for cand in (base + ".py", os.path.join(base, "__init__.py")):
            if os.path.isfile(cand):
                return cand
    return None


def _walk_file(path: str, is_core: bool, failures: list, warnings: list):
    """Blocklist-check imports + token grep + NotImplementedError scan.

    Returns the set of repo files imported by *path* (for one-level transitive
    expansion).
    """
    rel = os.path.relpath(path, REPO_ROOT)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError as exc:
        failures.append(f"{rel}: unreadable ({exc})")
        return set()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        failures.append(f"{rel}:{exc.lineno}: syntax error ({exc.msg})")
        return set()

    package = _package_of(path)
    referenced = set()
    for candidate, lineno in _import_candidates(tree, package):
        if module_name_violates(candidate):
            failures.append(
                f"{rel}:{lineno}: blocklisted import '{candidate}'")
        resolved = resolve_repo_module(candidate)
        if resolved is not None:
            referenced.add(os.path.abspath(resolved))

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            exc = node.exc
            name = None
            if isinstance(exc, ast.Name):
                name = exc.id
            elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                name = exc.func.id
            if name == "NotImplementedError":
                msg = f"{rel}:{node.lineno}: raise NotImplementedError"
                if is_core:
                    failures.append(msg + " in core eval file")
                else:
                    warnings.append(msg + " (transitive module)")

    for i, line in enumerate(source.splitlines(), start=1):
        m = WARN_TOKEN_RE.search(line)
        if m:
            warnings.append(f"{rel}:{i}: token '{m.group(0)}': {line.strip()[:120]}")
    return referenced


def run_static_check(run_eval_path: str, gems_dir: str) -> dict:
    failures: list = []
    warnings: list = []
    core_files = []
    if os.path.isfile(run_eval_path):
        core_files.append(os.path.abspath(run_eval_path))
    else:
        failures.append(f"core file missing: {run_eval_path}")
    if os.path.isdir(gems_dir):
        for root, _dirs, names in os.walk(gems_dir):
            if "__pycache__" in root:
                continue
            for name in sorted(names):
                if name.endswith(".py"):
                    core_files.append(os.path.abspath(os.path.join(root, name)))
    else:
        failures.append(f"core package missing: {gems_dir}")

    referenced = set()
    for path in core_files:
        referenced |= _walk_file(path, is_core=True,
                                 failures=failures, warnings=warnings)
    transitive = sorted(referenced - set(core_files))
    for path in transitive:
        _walk_file(path, is_core=False, failures=failures, warnings=warnings)

    return {
        "ok": not failures,
        "core_files": [os.path.relpath(p, REPO_ROOT) for p in core_files],
        "transitive_files": [os.path.relpath(p, REPO_ROOT) for p in transitive],
        "failures": failures,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# DYNAMIC check
# --------------------------------------------------------------------------

_SITECUSTOMIZE = """\
# GEMS audit hook: dump {module_name: __file__} at interpreter exit
# (D4 purity; the file path exposes bare-name imports from blocked dirs).
import atexit, json, os, sys

def _gems_audit_dump_modules():
    path = os.environ.get("GEMS_AUDIT_MODULES_JSON")
    if not path:
        return
    try:
        dump = {}
        for name, mod in list(sys.modules.items()):
            try:
                f = getattr(mod, "__file__", None)
            except Exception:
                f = None
            dump[name] = f if isinstance(f, str) else None
        with open(path, "w") as fh:
            json.dump(dump, fh)
    except Exception:
        pass

atexit.register(_gems_audit_dump_modules)
"""

_STRACE_OPEN_RE = re.compile(
    r'open(?:at2|at)?\((?:AT_FDCWD, )?"([^"]+)",\s*([A-Z0-9_|]+)[^)]*\)\s*=\s*(-?\d+)')


def parse_strace_reads(trace_path: str):
    """Successfully opened paths that were opened for reading."""
    reads = set()
    with open(trace_path, "r", errors="replace") as fh:
        for line in fh:
            m = _STRACE_OPEN_RE.search(line)
            if not m:
                continue
            path, flags, ret = m.group(1), m.group(2), int(m.group(3))
            if ret < 0:
                continue
            if "O_WRONLY" in flags:
                continue  # write-only opens are not reads
            reads.add(path)
    return sorted(reads)


def _allowed_read_roots(args, python_bin: str, hook_dir: str, scene_roots):
    home = os.path.expanduser("~")
    env_root = os.path.dirname(os.path.dirname(os.path.realpath(python_bin)))
    roots = [
        REPO_ROOT, env_root, os.path.realpath(os.getcwd()),
        os.path.realpath(args.out), os.path.realpath(hook_dir),
        os.path.dirname(os.path.realpath(args.checkpoint)),
        os.path.join(home, ".cache"), os.path.join(home, ".config"),
        os.path.join(home, ".local"), os.path.join(home, ".triton"),
        os.path.join(home, ".nv"), os.path.join(home, "micromamba"),
        tempfile.gettempdir(), "/tmp",
    ]
    roots.extend(_SYSTEM_ROOTS)
    roots.extend(scene_roots)
    return [os.path.realpath(r) for r in roots], env_root


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def audit_read_paths(reads, allowed_roots, env_root):
    """Blocklist failures + outside-allowlist warnings for traced reads.

    A read whose realpath is under <repo>/scripts/car_model/ FAILS
    unconditionally (checked on the realpath, so symlink indirection does not
    evade it), before any exemption.
    """
    failures, warnings = [], []
    system_like = [os.path.realpath(r) for r in _SYSTEM_ROOTS] + [env_root]
    for path in reads:
        rp = os.path.realpath(path)
        if _under(rp, SCRIPTS_CAR_MODEL_ROOT):
            failures.append(f"read path under scripts/car_model: {path}")
            continue
        if any(_under(rp, r) for r in system_like):
            continue  # OS / interpreter files: always allowed, never matched
        if BLOCKLIST_RE.search(_strip_stdlib_segments(rp, "/")):
            failures.append(f"read blocklisted path: {path}")
            continue
        if not any(_under(rp, r) for r in allowed_roots):
            warnings.append(f"read outside allowed roots: {path}")
    return failures, warnings


def audit_loaded_modules(modules, env_root=None):
    """Purity failures for a dynamic module dump (D4).

    ``modules`` is {module_name: __file__-or-None} (legacy list-of-names dumps
    are accepted). A module FAILS when:
      - its dotted NAME matches the blocklist (stdlib 'selectors' /
        '*.selector_events' segments exempted), OR
      - its ``__file__`` realpath is under <repo>/scripts/car_model/
        (regardless of name — catches bare-name imports after a
        sys.path.insert), OR
      - its ``__file__`` realpath matches the blocklist regex (same stdlib
        segment exemption; paths under the interpreter env / OS system roots
        are skipped for the path match — they are already covered by the
        name check).
    """
    if isinstance(modules, list):  # legacy dump format: names only
        modules = {name: None for name in modules}
    system_like = [os.path.realpath(r) for r in _SYSTEM_ROOTS]
    if env_root:
        system_like.append(os.path.realpath(env_root))
    failures = []
    for name in sorted(modules):
        mod_file = modules[name]
        if module_name_violates(name):
            failures.append(f"blocklisted module loaded: {name}")
            continue
        if not mod_file:
            continue
        rp = os.path.realpath(mod_file)
        if _under(rp, SCRIPTS_CAR_MODEL_ROOT):
            failures.append(
                f"module '{name}' loaded from scripts/car_model: {mod_file}")
            continue
        if any(_under(rp, r) for r in system_like):
            continue
        if BLOCKLIST_RE.search(_strip_stdlib_segments(rp, "/")):
            failures.append(
                f"module '{name}' file matches blocklist: {mod_file}")
    return failures


def _scene_data_roots(scene_name: str):
    """Declared data roots for the scene from tools/gems/scenes.py (best effort)."""
    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from tools.gems.scenes import SCENES  # data-only registry, D4-clean
    except Exception as exc:
        return [], f"scene registry unavailable ({exc}); scene data roots not allowlisted"
    spec = SCENES.get(scene_name)
    if spec is None:
        return [], f"scene '{scene_name}' not in registry"
    roots = [spec.source_path]
    for value in (spec.gt or {}).values():
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(item, str):
                roots.append(item if os.path.isdir(item) else os.path.dirname(item))
    return [r for r in roots if r], None


def run_dynamic_check(args, run_eval_path: str) -> dict:
    result = {"ok": False, "failures": [], "warnings": [], "notes": []}
    if not os.path.isfile(run_eval_path):
        result["failures"].append(f"run_eval.py not found at {run_eval_path}")
        return result

    python_bin = args.python or sys.executable
    hook_dir = tempfile.mkdtemp(prefix="gems_audit_hook_",
                                dir=os.path.realpath(args.out))
    with open(os.path.join(hook_dir, "sitecustomize.py"), "w") as fh:
        fh.write(_SITECUSTOMIZE)
    modules_json = os.path.join(hook_dir, "loaded_modules.json")
    trace_path = os.path.join(hook_dir, "strace_reads.txt")

    env = dict(os.environ)
    env["GEMS_AUDIT_MODULES_JSON"] = modules_json
    env["PYTHONPATH"] = hook_dir + os.pathsep + env.get("PYTHONPATH", "")

    eval_cmd = [python_bin, run_eval_path,
                "--checkpoint", args.checkpoint,
                "--scene", args.scene,
                "--out", args.out]
    if args.fast:
        eval_cmd += ["--skip-geometry", "--skip-downstream"]

    strace_bin = None if args.no_strace else shutil.which("strace")
    scene_roots, scene_note = _scene_data_roots(args.scene)
    if scene_note:
        result["notes"].append(scene_note)

    def _launch(with_strace: bool):
        cmd = list(eval_cmd)
        if with_strace:
            cmd = [strace_bin, "-f", "-qq", "-e", "trace=open,openat,openat2",
                   "-o", trace_path] + cmd
        return subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True,
                              timeout=args.timeout)

    used_strace = strace_bin is not None
    proc = _launch(with_strace=used_strace)
    if used_strace and proc.returncode != 0 and not os.path.exists(modules_json) \
            and (not os.path.exists(trace_path) or os.path.getsize(trace_path) == 0):
        # strace itself failed to attach (e.g. ptrace denied); retry untraced.
        result["notes"].append("strace could not attach; reran without tracing")
        used_strace = False
        proc = _launch(with_strace=False)

    result["returncode"] = proc.returncode
    result["command"] = " ".join(eval_cmd)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-15:])
        result["failures"].append(
            f"run_eval.py exited {proc.returncode}; stderr tail:\n{tail}")

    if os.path.exists(modules_json):
        with open(modules_json) as fh:
            modules = json.load(fh)
        result["n_modules_loaded"] = len(modules)
        env_root = os.path.dirname(os.path.dirname(os.path.realpath(python_bin)))
        result["failures"].extend(audit_loaded_modules(modules, env_root))
    else:
        result["failures"].append(
            "module dump missing (sitecustomize hook did not run) — cannot "
            "verify loaded modules")

    if used_strace and os.path.exists(trace_path):
        reads = parse_strace_reads(trace_path)
        allowed_roots, env_root = _allowed_read_roots(
            args, python_bin, hook_dir, scene_roots)
        read_failures, read_warnings = audit_read_paths(
            reads, allowed_roots, env_root)
        result["n_read_paths"] = len(reads)
        result["failures"].extend(read_failures)
        result["warnings"].extend(read_warnings)
    else:
        result["notes"].append(
            "strace unavailable — file-read audit skipped (best-effort check, "
            "not a failure per PROTOCOL)")

    result["ok"] = not result["failures"]
    return result


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="point_cloud_state_dict.pt to evaluate")
    parser.add_argument("--scene", required=True,
                        help="scene name from tools/gems/scenes.py")
    parser.add_argument("--out", required=True,
                        help="output dir (shared with run_eval.py)")
    parser.add_argument("--fast", action="store_true",
                        help="pass --skip-geometry --skip-downstream to run_eval.py")
    parser.add_argument("--python", default=None,
                        help="python interpreter for the eval subprocess "
                             "(default: this one)")
    parser.add_argument("--run-eval", default=os.path.join(REPO_ROOT, "run_eval.py"),
                        help="path to run_eval.py (default: repo root)")
    parser.add_argument("--no-strace", action="store_true",
                        help="skip the best-effort strace read audit")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="subprocess timeout in seconds")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gems_dir = os.path.join(REPO_ROOT, "tools", "gems")

    static = run_static_check(args.run_eval, gems_dir)
    dynamic = run_dynamic_check(args, args.run_eval)

    report = {
        "ok": static["ok"] and dynamic["ok"],
        "blocklist_pattern": BLOCKLIST_PATTERN,
        "static": static,
        "dynamic": dynamic,
    }
    report_path = os.path.join(args.out, "audit_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=1)

    for section in ("static", "dynamic"):
        for msg in report[section]["failures"]:
            print(f"FAIL [{section}] {msg}", file=sys.stderr)
        for msg in report[section]["warnings"]:
            print(f"WARN [{section}] {msg}")
        for msg in report[section].get("notes", []):
            print(f"NOTE [{section}] {msg}")

    if report["ok"]:
        print(f"AUDIT GREEN (report: {report_path})")
        return 0
    print(f"AUDIT RED (report: {report_path})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
