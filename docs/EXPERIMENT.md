# Experiment contract

## Main causal graph

```text
counterfactual visual self-state
    -> selected first action
    -> physical robot state after that action
    -> final task success
```

The repository separates these arrows with three intervention families.

1. **Pixel factorization:** change robot versus non-robot pixels while holding the physical state fixed.
2. **Action crossover:** query under one image but execute an action produced under another image.
3. **Reset rescue:** execute the stale action, restore the exact pre-action physical state, and continue with true observations.

## Frozen pair definition

A pair is admitted before outcomes are inspected only when:

- the nominal policy reaches the final task goal;
- a target subgoal becomes stably true;
- object-only advancement makes the target predicate true;
- another target candidate is not accidentally completed;
- hard reset plus action-prefix replay reproduces the trigger state;
- the endpoint and 25/50/75% donor states are finite and restorable.

## Primary endpoints

- final binary success;
- whether a previously false goal predicate is ever achieved;
- timeout;
- preservation of the externally completed predicate;
- first-action physical displacement;
- action-chunk distance to the true and stale queries.

## Invalidity policy

A runtime exception, replay mismatch, failed recomposition control, missing result, or checksum mismatch fails the inference gate. Invalid jobs are not silently counted as policy failures and are not replaced after the manifest is frozen.
