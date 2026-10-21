"""Step 8 experiment: does Engram help a tiny model remember multi-token names?

    uv run python -m engram.experiment            # about 3 to 4 minutes on a laptop CPU
    uv run python -m engram.experiment --quick    # about 1 minute, rougher numbers

Trains the same tiny transformer three or four ways on the synthetic world in
data.py, then probes the Engram model:
  1. backbone             2 layers, no Engram
  2. wider backbone       same, with a bigger MLP so it has as many parameters as 3
  3. backbone + Engram    Engram in layer 1
  4. Engram, no merging   same as 3 but without step 1's spelling merge
"""

import argparse

import numpy as np
import torch

from .compress import build_lookup
from .data import World, WorldConfig
from .engram import EngramConfig
from .model import ModelConfig, TinyLM
from .train import TrainConfig, evaluate, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--entities", type=int, default=1500)
    p.add_argument("--quick", action="store_true", help="600 steps, skip the no-merging run")
    p.add_argument("--rows", type=int, default=1009, help="minimum rows per hash head")
    p.add_argument("--heads", type=int, default=4, help="hash heads per N-gram order")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    steps = 600 if args.quick else args.steps

    world = World(WorldConfig(n_entities=args.entities, seed=args.seed))
    lookup, n_canonical = build_lookup(world.vocab)
    print(
        f"World: {args.entities} entities, vocab {world.vocab_size} tokens "
        f"-> {n_canonical} after merging spellings (step 1)\n"
    )

    def build(d_ff=256, engram=True, merge=True) -> TinyLM:
        torch.manual_seed(args.seed)
        cfg = ModelConfig(
            vocab_size=world.vocab_size,
            d_ff=d_ff,
            engram_layers=(1,) if engram else (),
            engram=EngramConfig(min_rows=args.rows, heads=args.heads),
        )
        return TinyLM(cfg, lookup if merge else None)

    engram_model = build()
    backbone = build(engram=False)
    extra = engram_model.n_params() - backbone.n_params()
    wide_ff = 256 + round(extra / (2 * 64 * 2))  # spend the same extra params on the MLPs instead
    runs = [
        ("backbone", backbone),
        ("wider backbone", build(d_ff=wide_ff, engram=False)),
        ("backbone + Engram", engram_model),
    ]
    if not args.quick:
        runs.append(("Engram, no merging", build(merge=False)))

    cfg = TrainConfig(steps=steps, eval_every=max(steps // 6, 1), seed=args.seed)
    results = {}
    for i, (name, model) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {name}  ({model.n_params() / 1e3:.0f}k params)")
        results[name] = train(model, world, cfg)
        print()

    print("Fact accuracy: how often the model predicts the word that follows an entity's name")
    checkpoints = [h["step"] for h in results["backbone"]]
    print(f"  {'model':<20} {'params':>7}  " + "".join(f"{'@' + str(s):>8}" for s in checkpoints))
    for name, model in runs:
        accs = "".join(f"{h['fact_acc']:8.1%}" for h in results[name])
        print(f"  {name:<20} {model.n_params() / 1e3:6.0f}k  {accs}")

    # Probe 1: knockout. Switch Engram off in the trained model.
    engram = engram_model.engrams()[0]
    on = evaluate(engram_model, world)
    engram.enabled = False
    off = evaluate(engram_model, world)
    engram.enabled = True
    print(f"\nKnockout: turn Engram off after training -> fact accuracy {on['fact_acc']:.1%} -> {off['fact_acc']:.1%}")

    # Probe 2: where does the gate open?
    ids, is_fact = world.sample(64, np.random.default_rng(7))
    engram_model.eval()
    with torch.no_grad():
        engram_model(ids[:, :-1])
    alpha = engram.last_alpha.squeeze(-1)
    name_end = is_fact[:, 1:]  # the next token is a fact, so this token ends a name
    kinds = np.array(world.kind)[ids[:, :-1].numpy()]
    by_kind = ", ".join(f"{k} words {alpha[torch.from_numpy(kinds == k)].mean():.2f}" for k in ("name", "fact", "filler"))
    print(f"Gate: mean alpha on {by_kind}")

    # Probe 3: one example, with alpha under each token.
    print("\nOne sample (alpha under each token, * = next token is a fact):")
    for start in range(1, 33, 8):
        span = range(start, start + 8)
        print("  " + "".join(f"{world.vocab[int(ids[0, j])].strip():>9}" for j in span))
        print("  " + "".join(f"{alpha[0, j]:9.2f}" for j in span))
        print("  " + "".join(f"{'*' if name_end[0, j] else '':>9}" for j in span))


if __name__ == "__main__":
    main()
