"""Check the course itself, not your code.

1. The reference solution (the `solutions` branch) passes every test.
2. With every core region blanked out, every test fails, so no test can be
   passed by accident.

    make verify
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pytest(cwd: Path) -> tuple[int, int]:
    """Run the tests in cwd against cwd's own code. Returns (passed, failed)."""
    src = os.pathsep.join(str(p) for p in sorted(cwd.glob("modules/*/src")))
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
        cwd=cwd, env=env, capture_output=True, text=True,
    ).stdout
    count = lambda word: sum(int(n) for n in re.findall(rf"(\d+) {word}", out.splitlines()[-1] if out else ""))
    return count("passed"), count("failed") + count("error")


def solutions_ref() -> str:
    """The local solutions branch if there is one, else the remote's (a fresh clone has only main)."""
    for ref in ("solutions", "origin/solutions"):
        if subprocess.run(["git", "rev-parse", "--verify", "-q", ref], cwd=ROOT, capture_output=True).returncode == 0:
            return ref
    sys.exit("No solutions branch found. Try: git fetch origin solutions")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "solutions"
        subprocess.run(["git", "worktree", "add", "--detach", str(tree), solutions_ref()], cwd=ROOT, check=True, capture_output=True)
        try:
            passed, failed = pytest(tree)
            print(f"reference solution: {passed} passed, {failed} failed")
            ok = failed == 0 and passed > 0
            subprocess.run([sys.executable, str(ROOT / "tools/make_stubs.py"), str(tree / "modules")], check=True, capture_output=True)
            passed_stubs, failed_stubs = pytest(tree)
            print(f"blank stubs:        {passed_stubs} passed, {failed_stubs} failed")
            ok = ok and passed_stubs == 0
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, capture_output=True)
    print("OK" if ok else "PROBLEM: see counts above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
