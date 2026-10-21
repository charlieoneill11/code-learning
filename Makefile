N ?= 1
STEP = $(wildcard modules/engram/steps/0$(N)_*)

.PHONY: step see test experiment verify

step:        ## run one step's tests:  make step N=3
	uv run pytest $(STEP)

see:         ## run your code on the toy example:  make see N=3
	uv run python -m engram.see $(N)

test:        ## run every test
	uv run pytest

experiment:  ## the step 8 experiment (about 3 minutes; QUICK=1 for about 1)
	uv run python -m engram.experiment $(if $(QUICK),--quick,)

verify:      ## check the course: reference passes, blank stubs fail
	uv run python tools/verify.py
