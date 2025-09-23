# videopoker

### Overview
- **Environment ID**: `videopoker`
- **Short description**: Single-turn Jacks or Better video poker environment that scores actions by their exact expected payout.
- **Tags**: games, single-turn, rl

### Datasets
- **Primary dataset(s)**: Synthetic Jacks or Better hands sampled from a standard 52-card deck when the environment loads.
- **Source links**: N/A (data is generated on the fly).
- **Split sizes**: `num_hands` prompts are generated for the train split (defaults to 200). Provide a custom evaluation dataset via `eval_dataset` if desired.

### Task
- **Type**: single-turn
- **Parser**: Default verifiers parser (no custom parsing required).
- **Rubric overview**: One reward function, `video_poker_reward`, enumerates every possible redraw outcome and returns the exact expected payout implied by the model's HOLD decision.

### Action format
Each prompt shows a five-card hand with zero-based indices and the paytable that defines the payouts. The model must reply in the format `HOLD: i j k`, listing the indices of the cards to keep in ascending order (omit indices to discard all cards). The environment discards every non-held card, evaluates every possible redraw from the remaining deck, and uses the average payout as the reward.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval videopoker
```

Configure model, sampling, and environment parameters:

```bash
uv run vf-eval videopoker \
  -m gpt-4.1-nano \
  -n 5 -r 1 -t 1024 -T 0.7 \
  -a '{"num_hands": 50, "seed": 42}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Provide your own `dataset` or `eval_dataset` via `--env-args` if you want to evaluate on a fixed set of hands.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_hands` | int | `200` | Number of synthetic prompts to generate when no dataset is supplied. |
| `seed` | int or null | `null` | Seed for reproducible hand generation. |
| `paytable` | mapping | Jacks or Better defaults | Optional mapping from hand categories (e.g., `"royal_flush"`) to payout values. |
| `dataset` | `datasets.Dataset` or null | `null` | Pre-built dataset to use for training/evaluation instead of synthetic generation. |
| `eval_dataset` | `datasets.Dataset` or null | `null` | Optional evaluation split. |
| `rubric` | `verifiers.rubrics.Rubric` or null | `null` | Custom rubric; defaults to the built-in expected-value scorer. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward equal to the expected payout of the HOLD decision. |
| `video_poker_reward` | Raw metric emitted by the rubric (identical to `reward`). |

### Programmatic usage

```python
from videopoker import load_environment

env = load_environment(num_hands=5, seed=0)
print(env.dataset[0]["prompt"])  # Inspect the first prompt
```

## Evaluation Reports
This section is reserved for auto-generated evaluation reports.
