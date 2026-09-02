# Pair-pack data contract

Each locked pair is stored as:

```text
pairs/<pair_id>/
├── pair.json
├── states.npz
└── SHA256SUMS.json
```

`pair.json` contains task identity, the four language variants, target object and predicate, source initial-state ID, and collection metadata.

`states.npz` contains:

- `initial_state`: LIBERO initial-state vector;
- `prefix_actions`: environment-space actions from reset to the trigger;
- `trigger_state`: state reproduced by reset plus prefix replay;
- `object_advanced_state`: trigger robot state with only the target object copied from the stable checkpoint;
- `endpoint_state`: first stable target-progress checkpoint on a successful nominal trajectory;
- `phase_25_state`, `phase_50_state`, `phase_75_state`: donor robot states along the active suffix.

The manifest builder validates SHA-256 checksums before freezing pair-condition jobs. Pair directories are immutable after `manifest.jsonl` is generated.
