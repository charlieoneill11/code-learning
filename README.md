# code-learning

Learn ML papers by writing their core ideas yourself, one small step at a
time, with a coding agent beside you. Each topic is a self-contained module
that ends with training a tiny model on your laptop's CPU.

## Start

```
git clone https://github.com/charlieoneill11/code-learning && cd code-learning
uv sync      # Python, PyTorch (CPU) and pytest, pinned by uv.lock
claude       # reads CLAUDE.md -> AGENTS.md; any agent that reads AGENTS.md works
```

Then say "next", or run `/next`.

## How a step works

1. Open the step's `STEP.md`, or ask the agent to walk you through it. It
   explains the idea with a toy example, so you don't need to have read the
   paper.
2. Write the code between `# >>> step N: your code` and `# <<<` in
   `modules/<module>/src/`. That's the core idea. The data, model, training
   loop and tests are already written.
3. `make see N=3` runs your code on the toy example. `make step N=3` runs the
   tests, and `/check` asks the agent to run them and review your code.
4. Ask the agent for a hint whenever you like.

## Modules

| Module | Paper | Steps | Ends with |
| --- | --- | --- | --- |
| [engram](modules/engram) | Conditional Memory via Scalable Lookup (DeepSeek, 2026) | 9 | A tiny CPU model learns 1,500 entity facts in 250 steps with Engram; a same-size transformer gets 3% |

## Commands

| Command | Does |
| --- | --- |
| `make step N=3` | Run one step's tests |
| `make see N=3` | Run your code on the toy example |
| `make experiment` | The Engram training experiment, about 3 minutes (`QUICK=1` for 1 minute) |
| `make test` | Every test |
| `make verify` | Check the course itself: the reference passes, blank stubs fail |

## Branches

`main` has the starter code, and your work goes there. `solutions` has the
reference. To peek, run `git diff main solutions -- <file>`, or ask the agent.
