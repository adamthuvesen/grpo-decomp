---
name: verify-grpo-decomp
description: "Drives the real grpo-decomp CLI and Modal launcher on CPU with committed fixtures. Use to run, launch, drive, or verify grpo-decomp, or to prove a change to it works."
---

# verify-grpo-decomp

One command proves a change works in the real entry points: the `grpo-decomp` CLI grades and
reports committed CompletionSets, and `modal_app.py` resolves the paid launch plan and refuses it.
Nothing reaches Modal, a GPU, Weights & Biases, or the Hugging Face Hub.

## Run

```bash
make verify                       # drives for files changed against main; the smoke drive if none match
make verify ARGS=all              # every drive, about 5 s
make verify ARGS=list             # each drive, what it proves, the paths it covers
make verify ARGS=control          # one drive with a wrong expectation; exits 0 only on FAIL
make verify ARGS="launch-plan"    # named drives
```

`make slow` is a different target: the full offline test gate. Run `make install` once first.

## Reading results

- Each drive prints `PASS|FAIL <drive> <seconds>s <what it proved or why it failed>`.
- A FAIL names the expected value and the observed value.
- Evidence is in `.verify/runs/<utc>-<sha7>[-dirty]/`: per drive, the argv, redacted env,
  exit code, stdout, and stderr, plus `launch-plan.json` and `summary.json`.
- The helper does not retry, so it never reports `FLAKY`. Run a failed drive again by name.

## Limits

- No drive trains. The locked CPU environment has no torch, TRL, or vLLM, and every
  real arm needs a Hugging Face model download. Training proof is a paid Modal run of
  `modal_app.py` with `--no-spawn --arm configs/correct.yaml --smoke-problems 8 --max-steps 5`.
  Run it only after Adam approves it. The refusal message prints the exact command.
- `generate` and `heldout` load a model. No drive covers them.

## Gotchas

- Import `modal_app` with the real `modal` package. The test suite fakes `modal`, and the
  fake once hid a crash in the `--approved` refusal under the real `LocalEntrypoint`.
- Report sets must be registered names (`dev`, `gsm8k-test`, ...). The drive copies the
  `mini` fixtures to `<arm>__dev`.
- The child environment points `MODAL_SERVER_URL` and `HF_ENDPOINT` at a dead loopback
  port, and a tripwire on `PATH` fails the run if anything shells out to `modal`, `wandb`, or `hf`.
