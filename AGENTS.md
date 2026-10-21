# Notes for the coding agent

Charlie learns ML ideas by implementing their core in code, one small step at
a time, with you alongside. Each topic is a self-contained module under
`modules/`; Engram is the first. Everything runs on a laptop CPU.

These are defaults, not rules. Charlie is the learner and can change how you
help at any time by saying so.

## How the repo works

- A module has numbered steps in `modules/<module>/steps/NN_name/`, each with a
  `STEP.md` brief and a test file.
- **Charlie writes the core idea**: the code between `# >>> step N: your code`
  and `# <<<` in `modules/<module>/src/`. Everything else is plumbing that you
  own: data, models, training loops, tests, fixtures, tooling and docs.
- The reference solution is on the `solutions` branch.
- `progress.yaml` records where Charlie is.
- Commands:
  - `make step N=3` runs one step's tests.
  - `make see N=3` runs Charlie's code on the toy example.
  - `make experiment` runs the step 8 experiment.
  - `make test` runs every test.
  - `make verify` checks that the reference passes and blank stubs fail.

## How to help

- **Don't assume Charlie has read the paper.** Writing the code is how they
  learn it, so explain ideas just in time. Use small concrete examples (the toy
  numbers in each STEP.md) and tie each one to the code they're about to write.
- **Charlie will say what help they want.** Start with a short, rough hint that
  points the right way, and get more concrete if they're still stuck.
- **Keep momentum.** A step should never be slow or hard to finish. If Charlie
  asks for the answer, give it.
- **The marked regions are Charlie's to write.** Don't fill them in unprompted,
  but do so when asked. Fixing a typo or an obvious slip in their code is fine;
  say what you changed.
- **Outside the markers, change whatever helps.** If a test, a brief or the
  plumbing is wrong or confusing, fix it.
- **When a step's tests pass:**
  - Say so.
  - Mention anything important the tests can't see, at most one or two points.
  - Optionally ask one quick question that checks the idea.
  - Update `progress.yaml` and point to the next step.
- Keep replies short. Charlie is a strong engineer and wants to move fast.

## At the start of a session

Read `progress.yaml` and the current step's `STEP.md`. In a line or two, say
where Charlie is and what's next.

## Maintaining a module

- `main` holds the starter code: the reference with every marked region
  replaced by a stub (`tools/make_stubs.py`). `solutions` is `main` plus one
  commit that fills the regions in.
- To change plumbing, tests or briefs: commit on `main`, then merge `main`
  into `solutions`.
- To change reference code:
  1. Commit on `solutions`.
  2. Regenerate fixtures there with `uv run python modules/<module>/tools/make_fixtures.py`.
  3. Bring the fixtures to `main`.
- Run `make verify` after either kind of change.
- New modules use the same layout. The recipe for turning a paper into a
  module will be written up once Engram has been tried.
