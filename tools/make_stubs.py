"""Blank out every core-idea region, leaving a stub in its place.

A region is everything between "# >>> step N: your code" and "# <<<". Run on
a copy of the reference code (tools/verify.py does this) to get the starter
code that lives on main:

    python tools/make_stubs.py modules/engram/src
"""

import re
import sys
from pathlib import Path

REGION = re.compile(
    r"^(?P<indent>[ \t]*)# >>> step (?P<n>\d+): your code\n.*?^(?P=indent)# <<<\n",
    re.MULTILINE | re.DOTALL,
)


def stub(match: re.Match) -> str:
    i, n = match["indent"], match["n"]
    return f'{i}# >>> step {n}: your code\n{i}raise NotImplementedError("step {n}")\n{i}# <<<\n'


def main(roots: list[str]) -> None:
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            text = path.read_text()
            new, count = REGION.subn(stub, text)
            if count:
                path.write_text(new)
                print(f"stubbed {count} region(s) in {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["modules"])
