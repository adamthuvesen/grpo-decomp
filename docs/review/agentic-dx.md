# Agentic development reassessment

Date: 2026-11-05. Baseline: `b424d2909152a0867b56b6c45d63d63a43d99ac2`.
Scope: local CPU development, artifact analysis, and testable GPU job planning.
Work is local on `adam/refactor-agentic-dx`. No changes were pushed.

## Verdict and score

**Excellent local agentic DX: seven of seven acceptance criteria met.**
This is an engineering assessment, not a measured benchmark or a claim of perfect software.
External execution readiness remains unverified where listed below.

| Criterion | Baseline assessment | Finished state and evidence |
| --- | --- | --- |
| Clear ownership | Good package boundary, competing script and adapter logic | One artifact publisher, attempt status owner, comparison constructor, grammar, and study workflow inventory |
| Valid states | Frozen schemas still accepted contradictory results and repeated seeds | Finite constrained records, derived-value checks, unique identities, and matched comparison protocols |
| Enforceable boundaries | Standalone imports and scientific comparability had gaps | Import-linter, isolated imports, installed-wheel smoke, validated comparisons, transactional plugin registration |
| Useful feedback | Duplicate inputs and missing dependencies could produce misleading outcomes | Contextual errors, clean CLI failures, full missing-input inventory, explicit external prerequisites |
| Behavioral correctness | Green baseline missed material failure cases | Historical reproduction, adversarial artifact and lifecycle cases, paired-statistic checks, actual CPU CLI execution |
| Ease of change | Study logic lived in unchecked scripts and repeated commands | Typed study modules, thin scripts, shared Make/CI gate, focused commands and ownership map |
| Independent judgment | Existing checks did not cover the discovered failure classes | Separate runtime, analysis, study, and final reviews, with demonstrated findings resolved |

The baseline passed 340 tests with five network tests deselected. That was useful coverage,
but it did not prevent the demonstrated failures below. Passing more tests alone is not the score.

## Evidence and intentional changes

- Repeating one battery directory three times previously produced three apparently independent seeds.
  Duplicate directories, labels, content, and known training identities now fail before inference.
- Same-ID problems with changed answers or questions could reach direct library analysis.
  Full ordered problem records now share one comparison boundary with CLI callers.
- Decoding drift, known prompt drift, and incompatible training protocols now fail before grading.
  Unequal base and trained sample counts remain supported. Decoding seeds may differ.
- Reusing a run directory could overwrite provenance while preserving an old final checkpoint.
  Each execution now owns a unique attempt directory and explicit terminal state.
- Failed attempts preserve successful attempts. Multiple successes require explicit selection.
  Attempt discovery uses authoritative status records, without a shared mutable completion index.
- New CompletionSets publish through staged atomic directories and integrity manifests.
  Legacy files remain readable. Conflicting writes fail, while identical retries preserve bytes.
- Reports reject non-finite values, impossible dimensions, and derived values that contradict observations.
  Student-t intervals retain their existing semantics and may extend beyond probability bounds.
- Coverage headlines no longer classify regressions as elicitation. Migration output exposes losses
  and uses descriptive language. Historical field names remain compatible.
- Esme's historical JSON remains byte-identical. The separate corrected analysis reports raw
  p-values of 0.03125 and Holm-adjusted values of 0.09375 for the three exploratory comparisons.
  Its intervals are marginal. It declares no retrospective primary endpoint.
- Persisted Esme and control p-values must agree with recorded seed effects where derivable.
  Supplying consistent-looking raw and adjusted p-values cannot fabricate significance.
- Countdown uses one bounded iterative evaluator with explicit task grammars.
  Excessive expressions fail grading without recursive evaluator exhaustion.
- Study scripts delegate to typed modules. The inventory includes the omitted placebo aggregations
  and identifies diagnostics whose original inputs are unavailable.

Historical context informed the work without becoming an invented incident log. Provenance
compatibility, unary Countdown grammar, restored caveats, and Platinum interpretation fixes remained
constraints. The Esme multiplicity correction addresses review feedback on PR 21. Other negative
cases are current defects or proactive architectural failure tests, not claimed historical incidents.

## Verification record

`make check` passed the complete locked offline gate. It includes:

- Ruff lint and format checks.
- Mypy over 54 source files.
- The one-way import contract and isolated harness imports without study or GPU dependencies.
- Offline tests, including an installed wheel in a fresh environment outside the checkout.
- Historical Esme byte reproduction and corrected scientific-value reproduction.
- Regenerated figure-content comparisons and documentation consistency.

The final suite contains 485 passing offline tests and six explicitly deselected external tests.
`make demo` exercised the real CLI and reproduced fixture strict accuracies of 1/3 and 1/2.
Those are fixture checks, not research estimates.

`make test-integration` failed as expected with actionable missing `torch` and `transformers` prerequisites.
`grpo-study validate` failed as expected and listed missing external CompletionSets and unavailable diagnostics.
Neither failure was converted into a skip or a successful verification claim.

Independent reviews found and resolved four substantive implementation issues:

1. A mutable completion alias could lose successful attempts across Modal Volume snapshots.
2. Plugin startup failures escaped CLI error handling.
3. Comparison logic used a requested model selector instead of the available resolved revision.
4. Persisted Esme p-values could contradict the recorded seed effects.

The final independent reassessment found no remaining demonstrated defects in its inspected scope.
Its focused run passed 89 tests. It did not independently review every changed line.

## External limits and release criteria

The local score does not establish these claims:

- Live transformers generation and Hugging Face dataset access work in the intended environment.
- Real CUDA training, vLLM generation, model cleanup, and final-checkpoint evaluation work together.
- Modal deployment and Volume persistence work across actual container lifetimes.
- Hosted Linux/macOS and Python 3.13/3.14 CI jobs pass. The matrix is configured but was not pushed or run.
- Missing original CompletionSets reproduce every published aggregate.
- Unpinned external training models have verified resolved revision identity.

Before a GPU release, run the explicit integration and GPU checks in `docs/development.md`,
then verify a bounded training attempt and held-out curve on the intended deployment.
Record actual hardware, versions, inputs, outputs, and any skipped verification.
Local filesystem tests do not establish Modal's runtime behavior.

Stop local iteration when the gate passes, the negative cases fail correctly, and review has no
unresolved demonstrated defects. Reopen the assessment when new behavior or external evidence changes those conditions.
