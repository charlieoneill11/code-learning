---
name: check
description: Run the current step's tests and give feedback. Use when Charlie says they're done, asks you to check their code, or runs /check.
---

1. Work out the step. Use the one Charlie names, or else `current_step` in `progress.yaml`.
2. Run `make step N=<n>`.
3. **If tests fail:**
   - Say which test failed and what it checks, in plain words.
   - Give a rough hint toward the cause.
   - Don't rewrite Charlie's code unless they ask.
4. **If tests pass:**
   - Say so.
   - Add one or two notes on anything important the tests can't see, only if it matters.
   - Optionally ask one quick question about the idea.
   - In `progress.yaml`, add the step to `done` with today's date and set `current_step` to the next step.
   - Point to the next step.
