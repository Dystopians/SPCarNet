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

# Stage-4 ECR mode (PROTOCOL 1.2.0 §4E): the evidence-transport module
# (utils/evidence_lumigraph_adapter.py) and tools/ecr are LEGAL render-time
# inputs — train-view evidence is part of the shipped artifact. Everything
# else stays dead: teacher/selector/ecsr_/car_model remain blocked. The
# no-test-GT guarantee moves to the ECR-specific checks below (transport
# reads confined to the cache manifest; manifest disjoint from the test
# split; frozen per-view kwargs; no `original_image` token in the transport
# module).
ECR_BLOCKLIST_PATTERN = (
    r"(teacher|ecsr_|phasej|phase_j|selector|calibrator|"
    r"arbitrat|scripts[./\\]car_model|car_model)"
)
ECR_BLOCKLIST_RE = re.compile(ECR_BLOCKLIST_PATTERN)

# The regex the current invocation enforces (set in main; default Stage-2).
ACTIVE_BLOCKLIST_RE = BLOCKLIST_RE

# Directory of the Stage-4 ECR render-path package. In Stage-2 (default)
# mode the STATIC walk does not expand into it (run_eval.py's ecr import is
# mode-gated and never executes with --renderer base); instead the DYNAMIC
# check FAILS if any tools.ecr / transport module actually loads in base
# mode — the guarantee is enforced at runtime, where it is real.
TOOLS_ECR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "tools", "ecr"))

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
    return bool(ACTIVE_BLOCKLIST_RE.search(_strip_stdlib_segments(name, ".")))


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


def run_static_check(run_eval_path: str, gems_dir: str,
                     ecr_mode: bool = False) -> dict:
    failures: list = []
    warnings: list = []
    notes: list = []
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

    if ecr_mode:
        # ECR mode: the transport render path is CORE and must satisfy the
        # ECR blocklist. build_cache.py is the offline train-side builder
        # (it legitimately reads TRAIN images via cam.original_image); the
        # `original_image` token check applies to every OTHER tools/ecr
        # file, structurally proving the render path never touches an
        # object that can carry test GT.
        if os.path.isdir(TOOLS_ECR_ROOT):
            for root, _dirs, names in os.walk(TOOLS_ECR_ROOT):
                if "__pycache__" in root:
                    continue
                for name in sorted(names):
                    if name.endswith(".py"):
                        core_files.append(
                            os.path.abspath(os.path.join(root, name)))
        else:
            failures.append(f"core package missing: {TOOLS_ECR_ROOT}")

    referenced = set()
    for path in core_files:
        referenced |= _walk_file(path, is_core=True,
                                 failures=failures, warnings=warnings)
    if not ecr_mode:
        # Stage-2 mode: do NOT expand into tools/ecr — run_eval.py's ecr
        # import is mode-gated (never executes with --renderer base); the
        # dynamic check fails the audit if any tools.ecr / transport module
        # actually loads in base mode (PROTOCOL 1.2.0 changelog).
        deferred = sorted(p for p in referenced
                          if _under(os.path.realpath(p), TOOLS_ECR_ROOT))
        if deferred:
            notes.append(
                "Stage-4 ecr modules referenced but not expanded (mode-gated "
                "import; base-mode non-loading enforced dynamically): "
                + ", ".join(os.path.relpath(p, REPO_ROOT) for p in deferred))
        referenced = {p for p in referenced
                      if not _under(os.path.realpath(p), TOOLS_ECR_ROOT)}
    transitive = sorted(referenced - set(core_files))
    for path in transitive:
        _walk_file(path, is_core=False, failures=failures, warnings=warnings)

    if ecr_mode:
        for path in core_files:
            rp = os.path.realpath(path)
            if not _under(rp, TOOLS_ECR_ROOT):
                continue
            if os.path.basename(path) == "build_cache.py":
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh, start=1):
                        if "original_image" in line:
                            failures.append(
                                f"{os.path.relpath(path, REPO_ROOT)}:{i}: "
                                "'original_image' token in ECR render path "
                                "(test-GT accessor forbidden)")
            except OSError as exc:
                failures.append(f"{os.path.relpath(path, REPO_ROOT)}: "
                                f"unreadable ({exc})")

    return {
        "ok": not failures,
        "mode": "ecr" if ecr_mode else "stage2",
        "core_files": [os.path.relpath(p, REPO_ROOT) for p in core_files],
        "transitive_files": [os.path.relpath(p, REPO_ROOT) for p in transitive],
        "failures": failures,
        "warnings": warnings,
        "notes": notes,
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
        if ACTIVE_BLOCKLIST_RE.search(_strip_stdlib_segments(rp, "/")):
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
        if ACTIVE_BLOCKLIST_RE.search(_strip_stdlib_segments(rp, "/")):
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
    if getattr(args, "ecr", False):
        eval_cmd += ["--renderer", "ecr", "--ecr-cache", args.ecr_cache]

    strace_bin = None if args.no_strace else shutil.which("strace")
    scene_roots, scene_note = _scene_data_roots(args.scene)
    if scene_note:
        result["notes"].append(scene_note)
    if getattr(args, "ecr", False) and args.ecr_cache:
        scene_roots = list(scene_roots) + [os.path.realpath(args.ecr_cache)]

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
        if not getattr(args, "ecr", False):
            # Stage-2 mode strengthening (PROTOCOL 1.2.0): base-mode eval
            # must never load the Stage-4 ECR render path or the transport
            # module — this is what makes the static walker's tools/ecr
            # non-expansion sound.
            mod_map = ({name: None for name in modules}
                       if isinstance(modules, list) else modules)
            for name in sorted(mod_map):
                mod_file = mod_map[name]
                under_ecr = bool(
                    mod_file
                    and _under(os.path.realpath(mod_file), TOOLS_ECR_ROOT))
                if name == "tools.ecr" or name.startswith("tools.ecr.") \
                        or under_ecr:
                    result["failures"].append(
                        f"base-mode eval loaded Stage-4 ecr module: {name} "
                        f"({mod_file})")
    else:
        result["failures"].append(
            "module dump missing (sitecustomize hook did not run) — cannot "
            "verify loaded modules")

    if getattr(args, "ecr", False):
        ecr_failures, ecr_notes = _ecr_specific_checks(args)
        result["failures"].extend(ecr_failures)
        result["notes"].extend(ecr_notes)

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


def _ecr_specific_checks(args):
    """PROTOCOL 1.2.0 §4E: prove no test-GT dependency and no per-test-view
    parameter injection in the ECR render path.

    1. every file the transport read is listed in the cache manifest
       (run_eval's confined-loader read log, cross-checked here);
    2. the manifest's train_views are DISJOINT from the scene's test split
       (recomputed independently from the registry, not trusted from the
       manifest);
    3. the manifest was built for exactly the audited checkpoint;
    4. the transport kwargs hash is identical for every test view (frozen
       config; no per-view injection).
    """
    failures, notes = [], []
    out = os.path.realpath(args.out)
    cache_root = os.path.realpath(args.ecr_cache)

    reads_path = os.path.join(out, "ecr_transport_reads.json")
    if not os.path.exists(reads_path):
        failures.append(f"ecr transport read log missing: {reads_path}")
    else:
        with open(reads_path) as fh:
            reads = json.load(fh)
        if reads.get("all_reads_in_manifest") is True:
            notes.append(
                f"ecr transport reads: {reads.get('n_reads')} files, all "
                f"within the cache manifest ({reads.get('n_manifest_files')} "
                "files)")
        else:
            head = (reads.get("reads_outside_manifest") or [])[:5]
            failures.append(
                f"ecr transport read files OUTSIDE the cache manifest: {head}")

    manifest_path = os.path.join(cache_root, "manifest.json")
    if not os.path.exists(manifest_path):
        failures.append(f"ecr cache manifest missing: {manifest_path}")
        return failures, notes
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from tools.gems.scenes import SCENES
        from tools.gems.eval_context import _read_scene_info
        spec = SCENES[args.scene]
        info = _read_scene_info(spec)
        test_names = {str(c.image_name) for c in info.test_cameras}
    except Exception as exc:
        failures.append(f"could not recompute the test split for the "
                        f"manifest-disjointness check: {exc}")
    else:
        train_views = set(manifest.get("train_views", []))
        overlap = sorted(test_names & train_views)
        if overlap:
            failures.append(
                f"cache manifest lists TEST view names (D4 violation): "
                f"{overlap[:5]}")
        else:
            notes.append(
                f"cache manifest train views ({len(train_views)}) disjoint "
                f"from the recomputed test split ({len(test_names)})")

    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "point_cloud_state_dict.pt")
    import hashlib
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as fh:
        h.update(fh.read(16 * 1024 * 1024))
    manifest_sha = (manifest.get("checkpoint") or {}).get("sha256_first16mb")
    if manifest_sha != h.hexdigest():
        failures.append(
            f"cache manifest checkpoint fingerprint {manifest_sha} != audited "
            f"checkpoint {h.hexdigest()}")

    metrics_path = os.path.join(out, "metrics.json")
    if not os.path.exists(metrics_path):
        failures.append(f"metrics.json missing: {metrics_path}")
    else:
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        if metrics.get("renderer") != "ecr":
            failures.append(
                f"metrics.json renderer={metrics.get('renderer')!r}, "
                "expected 'ecr'")
        ecr_block = metrics.get("ecr") or {}
        if ecr_block.get("per_view_kwargs_identical") is True:
            notes.append(
                "transport kwargs hash identical across all test views "
                f"(config_hash {str(ecr_block.get('config_hash'))[:12]}) — "
                "no per-test-view parameter injection")
        else:
            failures.append(
                "per-view transport kwargs are NOT identical (or missing) — "
                "possible per-test-view parameter injection")
    return failures, notes


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
    parser.add_argument("--ecr", action="store_true",
                        help="Stage-4 ECR audit mode (PROTOCOL 1.2.0 §4E): "
                             "runs run_eval.py --renderer ecr and proves the "
                             "no-test-GT / frozen-config guarantees for the "
                             "evidence-transport path")
    parser.add_argument("--ecr-cache", default=None,
                        help="evidence cache dir (required with --ecr)")
    args = parser.parse_args()

    if args.ecr and not args.ecr_cache:
        parser.error("--ecr requires --ecr-cache")
    global ACTIVE_BLOCKLIST_RE
    ACTIVE_BLOCKLIST_RE = ECR_BLOCKLIST_RE if args.ecr else BLOCKLIST_RE

    os.makedirs(args.out, exist_ok=True)
    gems_dir = os.path.join(REPO_ROOT, "tools", "gems")

    static = run_static_check(args.run_eval, gems_dir, ecr_mode=args.ecr)
    dynamic = run_dynamic_check(args, args.run_eval)

    report = {
        "ok": static["ok"] and dynamic["ok"],
        "mode": "ecr" if args.ecr else "stage2",
        "blocklist_pattern": (ECR_BLOCKLIST_PATTERN if args.ecr
                              else BLOCKLIST_PATTERN),
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
