#!/usr/bin/env python3
"""Local pre-flight validation for autoresearch-gaudi.

Runs on your laptop (no HPU needed) to catch issues before spending
cloud credits. Validates syntax, imports, Docker build readiness,
file integrity, and configuration consistency.

Usage:
    python scripts/preflight.py
"""

import ast
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

PASS = 0
FAIL = 0
WARN = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  \033[32mOK\033[0m  {msg}")


def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  \033[31mFAIL\033[0m {msg}")


def warn(msg):
    global WARN
    WARN += 1
    print(f"  \033[33mWARN\033[0m {msg}")


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 1. Syntax check all Python files ──────────────────────────

def check_syntax():
    section("1. Python syntax")
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))
    py_files = [f for f in py_files if ".venv" not in str(f) and "__pycache__" not in str(f)]

    for f in py_files:
        rel = f.relative_to(PROJECT_ROOT)
        try:
            with open(f) as fh:
                compile(fh.read(), str(rel), "exec")
            ok(str(rel))
        except SyntaxError as e:
            fail(f"{rel}: {e}")


# ── 2. Critical file existence ────────────────────────────────

def check_files():
    section("2. Required files")
    required = [
        "train_gaudi.py", "prepare.py", "run_suite.py",
        "Dockerfile", "docker-compose.yml", "pyproject.toml",
        "backends/__init__.py", "backends/muon_gaudi.py",
        "tui/__init__.py", "tui/orchestrator.py", "tui/headless.py",
        "tui/llm_backend.py", "tui/parser.py", "tui/results.py",
        "tui/hardware.py", "tui/credentials.py", "tui/git_manager.py",
        "scripts/verify_hpu.py", "scripts/benchmark_hpu.py",
        "program.md", "README.md", ".gitignore",
    ]
    for f in required:
        if (PROJECT_ROOT / f).exists():
            ok(f)
        else:
            fail(f"MISSING: {f}")


# ── 3. Dockerfile integrity ──────────────────────────────────

def check_dockerfile():
    section("3. Dockerfile")
    df = (PROJECT_ROOT / "Dockerfile").read_text()

    if "vault.habana.ai" in df:
        ok("Base image: Habana official")
    else:
        fail("Base image not from vault.habana.ai")

    # Check if any pip install line actually installs torch as a package
    # (ignore comments and package names that contain "torch" as substring like "pytorch")
    pip_installs_torch = False
    for line in df.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "pip install" in stripped:
            # Check for bare "torch" as a package (not pytorch, not tiktoken)
            packages = stripped.split("pip install")[-1]
            tokens = re.split(r'[\s\\]+', packages)
            for tok in tokens:
                pkg_name = re.split(r'[><=!]', tok)[0].strip()
                if pkg_name == "torch" or pkg_name == "pytorch":
                    pip_installs_torch = True
    if pip_installs_torch:
        fail("Dockerfile pip-installs torch — this will BREAK the HPU build")
    else:
        ok("Does not pip install torch (correct)")

    # Check required pip packages
    for pkg in ["numpy", "pyarrow", "requests", "rustbpe", "tiktoken", "anthropic", "textual"]:
        if pkg in df:
            ok(f"Installs {pkg}")
        else:
            warn(f"Missing {pkg} in Dockerfile")

    if "COPY . ." in df or "COPY ./ ./" in df:
        ok("Copies source code")
    else:
        fail("No COPY . . in Dockerfile")


# ── 4. docker-compose.yml sanity ─────────────────────────────

def check_compose():
    section("4. docker-compose.yml")
    dc = (PROJECT_ROOT / "docker-compose.yml").read_text()

    for svc in ["train", "prepare", "agent", "verify"]:
        if f"{svc}:" in dc:
            ok(f"Service: {svc}")
        else:
            fail(f"Missing service: {svc}")

    if "runtime: habana" in dc:
        ok("Habana runtime configured")
    else:
        fail("Missing runtime: habana")

    if "HABANA_VISIBLE_DEVICES=all" in dc:
        ok("HABANA_VISIBLE_DEVICES=all set")
    else:
        warn("HABANA_VISIBLE_DEVICES not set")

    if "ANTHROPIC_API_KEY" in dc:
        ok("ANTHROPIC_API_KEY passthrough in agent service")
    else:
        fail("ANTHROPIC_API_KEY not configured for agent service")

    if "ipc: host" in dc:
        ok("IPC host mode")
    else:
        warn("IPC host not set (may affect shared memory)")


# ── 5. Training script consistency ───────────────────────────

def check_training_script():
    section("5. train_gaudi.py")
    src = (PROJECT_ROOT / "train_gaudi.py").read_text()

    # HP block markers (orchestrator depends on these)
    if "# Hyperparameters" in src and "# Setup" in src:
        ok("HP block markers present (orchestrator can parse)")
    else:
        fail("Missing HP block markers — orchestrator will crash")

    if 'torch.device("hpu")' in src or "torch.device('hpu')" in src:
        ok("Device: hpu")
    else:
        fail("No torch.device('hpu') found")

    if 'backend="hpu_backend"' in src:
        ok("torch.compile(backend='hpu_backend')")
    else:
        warn("No hpu_backend compilation found")

    if "habana_frameworks" in src:
        ok("Imports habana_frameworks")
    else:
        warn("No habana_frameworks import")

    if "evaluate_bpb" in src:
        ok("Calls evaluate_bpb (from prepare.py)")
    else:
        fail("Missing evaluate_bpb call")

    # Check output format (parser depends on this)
    for field in ["val_bpb:", "training_seconds:", "peak_vram_mb:", "mfu_percent:", "depth:", "chip:"]:
        if field in src:
            ok(f"Output field: {field}")
        else:
            fail(f"Missing output field: {field} (parser will fail)")


# ── 6. Parser compatibility ──────────────────────────────────

def check_parser():
    section("6. Output parser")
    parser_src = (PROJECT_ROOT / "tui" / "parser.py").read_text()

    # Check it can parse the expected output format
    if "val_bpb" in parser_src:
        ok("Parser handles val_bpb")
    else:
        fail("Parser doesn't look for val_bpb")

    if "FinalMetrics" in parser_src:
        ok("FinalMetrics dataclass present")
    else:
        fail("Missing FinalMetrics")


# ── 7. Orchestrator configuration ────────────────────────────

def check_orchestrator():
    section("7. Orchestrator & headless")
    orch_src = (PROJECT_ROOT / "tui" / "orchestrator.py").read_text()
    head_src = (PROJECT_ROOT / "tui" / "headless.py").read_text()

    if "train_gaudi.py" in orch_src or "train_gaudi.py" in head_src:
        ok("Default training script: train_gaudi.py")
    else:
        warn("Default training script may not be train_gaudi.py")

    if "gpu_name" in orch_src:
        ok("Deployment fencing: gpu_name in results")
    else:
        warn("No gpu_name in orchestrator (deployment fencing missing)")

    if "HP_BLOCK_START" in orch_src and "HP_BLOCK_END" in orch_src:
        ok("HP block extraction markers configured")
    else:
        fail("Missing HP block markers in orchestrator")


# ── 8. Data pipeline ─────────────────────────────────────────

def check_data():
    section("8. Data pipeline (prepare.py)")
    src = (PROJECT_ROOT / "prepare.py").read_text()

    if "climbmix-400b-shuffle" in src:
        ok("ClimbMix dataset URL configured")
    else:
        fail("ClimbMix URL missing")

    if "spawn" in src:
        ok("Spawn-safe multiprocessing (required for Gaudi)")
    else:
        warn("No spawn context — may deadlock on Gaudi")

    if "shard_06542" in src or "VAL_SHARD = 6542" in src or "MAX_SHARD = 6542" in src:
        ok("Validation shard pinned (shard_06542)")
    else:
        fail("Validation shard not pinned")

    if "VOCAB_SIZE = 8192" in src:
        ok("Vocab size: 8192")
    else:
        warn("Unexpected vocab size")


# ── 9. Git state ─────────────────────────────────────────────

def check_git():
    section("9. Git repository")
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if r.returncode == 0:
            ok("Git repo valid")
            dirty = [l for l in r.stdout.strip().split("\n") if l.strip()]
            if dirty:
                warn(f"{len(dirty)} uncommitted changes")
            else:
                ok("Working tree clean")
        else:
            fail("Not a git repository")
    except FileNotFoundError:
        warn("git not found")

    # Check remote
    try:
        r = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
        if "autoresearch-gaudi" in r.stdout:
            ok("Remote: autoresearch-gaudi")
        else:
            warn("Remote may not point to autoresearch-gaudi")
    except Exception:
        pass


# ── 10. Gitignore for results ────────────────────────────────

def check_gitignore():
    section("10. Deployment fencing (.gitignore)")
    gi = (PROJECT_ROOT / ".gitignore").read_text()

    if "/results.tsv" in gi:
        ok("Root results.tsv gitignored")
    else:
        fail("results.tsv not gitignored — cross-GPU contamination risk")

    if "results/**/results.tsv" in gi:
        ok("Per-dataset results gitignored")
    else:
        fail("Per-dataset results not gitignored")

    if "manifest.json" in gi:
        ok("manifest.json gitignored")
    else:
        warn("manifest.json not in gitignore")


# ── 11. LLM backend ─────────────────────────────────────────

def check_llm():
    section("11. LLM backend (agent)")
    src = (PROJECT_ROOT / "tui" / "llm_backend.py").read_text()

    if "anthropic" in src:
        ok("Anthropic SDK referenced")
    else:
        fail("No anthropic import")

    if "Gaudi" in src or "gaudi" in src or "HPU" in src:
        ok("System prompt references Gaudi/HPU")
    else:
        warn("System prompt may not mention Gaudi 3")

    # Check ANTHROPIC_API_KEY is expected
    if "ANTHROPIC_API_KEY" in src:
        ok("Reads ANTHROPIC_API_KEY")
    else:
        warn("May not read ANTHROPIC_API_KEY from env")


# ── 12. Launch script ────────────────────────────────────────

def check_launch():
    section("12. Launch automation")
    launch = PROJECT_ROOT / "scripts" / "launch.sh"
    if launch.exists():
        ok("scripts/launch.sh exists")
        src = launch.read_text()
        for check in ["docker compose build", "verify_hpu", "prepare", "ANTHROPIC_API_KEY"]:
            if check in src:
                ok(f"Launch script includes: {check}")
            else:
                warn(f"Launch script missing: {check}")
    else:
        warn("scripts/launch.sh not found — create it for launch day")


# ── Summary ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  autoresearch-gaudi  Pre-Flight Check")
    print("=" * 60)

    check_syntax()
    check_files()
    check_dockerfile()
    check_compose()
    check_training_script()
    check_parser()
    check_orchestrator()
    check_data()
    check_git()
    check_gitignore()
    check_llm()
    check_launch()

    print(f"\n{'=' * 60}")
    if FAIL == 0:
        print(f"  \033[32mPREFLIGHT PASSED\033[0m — {PASS} checks OK, {WARN} warnings")
        print(f"  Ready for deployment on Gaudi 3.")
    else:
        print(f"  \033[31mPREFLIGHT FAILED\033[0m — {FAIL} failures, {WARN} warnings, {PASS} OK")
        print(f"  Fix failures before deploying.")
    print(f"{'=' * 60}\n")

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
