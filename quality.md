# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Read the in-scope files**: The repo is small enough to read the relevant approximation code for full context:
   - `README.md` — repository context.
   - `locomotif/locomotif.py` — motif-set loop and high-level orchestration.
   - `locomotif/loco.py` — LoCo wrapper, tau estimation, cumulative similarity entrypoints.
   - `locomotif/loco_jit.py` — core kernels, path extraction, sorting, pruning.
   - `/Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling/compact_profiling_all_quality.py` — benchmark harness. Do not modify.
2. **Verify the profiling workspace exists**: Check that the sister directory `/Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling` exists and that `cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --max-cases 100` runs there. If not, tell the human what is missing.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is benchmarked on the profiling workspace. The primary benchmark is:

`cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --max-cases 100`

The `--max-cases 100` baseline must be re-established before comparing experiments.

**What you CAN do:**
- Modify `locomotif/locomotif.py`, `locomotif/loco.py`, and `locomotif/loco_jit.py`.
- Use `main` as the style and structure reference when deciding how code should look after cleanup. Prefer changes that reduce drift from `main` and make the branch easier to audit.
- Simplify algorithms, control flow, data flow, and helper structure when the resulting behavior is clearer and benchmark quality is preserved or improved.
- Remove redundant branches, duplicated logic, unnecessary state, and incidental complexity.
- Tighten motif quality, path quality, pruning correctness, tau estimation, and approximation behavior when the changes remain faithful to the broad LoCoMotif semantics.
- Keep runtime at least as good as baseline while simplifying the code. Runtime may stay the same or improve, but it must not get worse.

**What you CANNOT do:**
- Modify the profiling harness in `/Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling`.
- Hardcode per-dataset behavior or tune special-case hyperparameters for individual datasets.
- Disable warping support.
- Replace LoCoMotif with a fundamentally different method that no longer matches the broad semantics of the original algorithm.
- Add complexity whose main purpose is squeezing out runtime at the expense of clarity or auditability.

**Algorithmic reference**: The implementation may change substantially, but it should still preserve the broad LoCoMotif semantics from the paper at `/Users/fre/Documents/University/2025-2026/thesis/papers/LoCoMotif_discovering_time-warped_motifs_in_time_s.pdf`. Use that paper as the reference for what must remain true at a high level, rather than trying to preserve exact code behavior.

**Auditability criterion**: Prefer code that looks like it could plausibly live on `main`. Favor direct data flow, fewer moving parts, obvious invariants, and changes that are easy to review in a diff against `main`.

**Do not preserve exact paths for their own sake**: The discovered paths, motif sets, and internal behavior do not need to stay identical to the current implementation. Optimize for motif quality, correctness, and simpler structure, while keeping the broad LoCoMotif semantics.

**Large-series safety**: Assume future runs may use very large time series. Avoid changes that risk integer overflow, indexing overflow, or size-dependent corruption on much larger inputs.

**The goal is simple: improve motif quality, or preserve it while clearly simplifying the code, and never accept a runtime increase on the full `--max-cases 100` benchmark.** Favor structural cleanups that make the implementation easier to audit and maintain, but only keep them when runtime does not increase.

**Quality-and-simplicity criterion**: Keep a change only if average quality improves, or if average quality is unchanged and the code is meaningfully simpler, clearer, or closer to `main`. If quality drops, discard it. If quality is unchanged but the code is not clearly simpler, discard it.

**Simplicity criterion**: All else being equal, simpler is better. Prefer deleting machinery over adding it. An unchanged-quality change is only valuable when the simplification is obvious in the diff and reduces real complexity.

**Runtime rule**: Runtime is a hard gate, not a soft guideline. Do not keep any change whose full-run runtime is higher than baseline. Equal runtime is acceptable. Lower runtime is better.

**The first run**: Your very first run should always be to establish the baseline, so you will run the full benchmark script as is.

## Output format

Once the script finishes it prints per-dataset summaries and a global summary like this:

```
========================================================
SUMMARY ALL DATASETS
========================================================
articularywordrecognition        | quality=0.7187
...
average_quality_score=0.6174

========================================================
RUNTIME PROFILE
========================================================
total_seconds=194.174353
```

You can extract the key metrics from the log file:

```
grep "^average_quality_score=\|^total_seconds=" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 9 columns:

```
commit	avg_quality	total_seconds	cricket100_quality	cricket100_seconds	mallat100_quality	mallat100_seconds	status	description
```

1. git commit hash (short, 7 chars) or `wip`
2. average quality on the full `--max-cases 100` benchmark
3. total runtime on the full `--max-cases 100` benchmark
4. quality on `--benchmark cricket --max-cases 100`
5. runtime on `--benchmark cricket --max-cases 100`
6. quality on `--benchmark mallat --max-cases 100`
7. runtime on `--benchmark mallat --max-cases 100`
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:

```
commit	avg_quality	total_seconds	cricket100_quality	cricket100_seconds	mallat100_quality	mallat100_seconds	status	description
a1b2c3d	0.6174	194.17	0.8136	138.14	0.7546	31.03	keep	baseline
b2c3d4e	0.6210	186.50	0.8200	129.00	0.7600	28.80	keep	simplify tau estimate while preserving output quality
c3d4e5f	0.6120	175.10	0.7900	120.00	0.7400	26.00	discard	remove pruning branch but quality regressed
d4e5f6g	0.0000	0.00	0.0000	0.00	0.0000	0.00	crash	broken cleanup of path bookkeeping
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar21`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Pick one cleanup or quality idea that should make the code simpler, clearer, or closer to `main` while preserving or improving benchmark quality.
3. Tune the approximation code with that one idea by directly hacking the code.
4. Run the screening benchmarks:
   - `cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --benchmark cricket --max-cases 100 > run_cricket.log 2>&1`
   - `cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --benchmark mallat --max-cases 100 > run_mallat.log 2>&1`
5. Read out the screening results:
   - `grep "^average_quality_score=\|^total_seconds=" run_cricket.log`
   - `grep "^average_quality_score=\|^total_seconds=" run_mallat.log`
6. If either screening run crashes, run `tail -n 50` on the corresponding log, decide whether the issue is fixable, and either retry or discard.
7. If screening shows any quality loss, or any runtime increase without a clear path to meeting the final keep rule, discard immediately and revert to where you started.
8. If screening looks promising, run the promotion benchmark:
   - `cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --max-cases 100 > run.log 2>&1`
9. Read out the promotion results:
   - `grep "^average_quality_score=\|^total_seconds=" run.log`
10. Record the results in the TSV.
11. Promotion rule:
   - Keep if full-run average quality improves and runtime is less than or equal to baseline.
   - Keep if full-run average quality is unchanged, runtime is less than or equal to baseline, and the code is materially simpler or closer to `main`.
   - If full-run average quality improves and the code also becomes simpler, keep.
   - Otherwise discard.
12. If the change is kept, git commit it and advance the branch.
13. If the change is discarded, revert only the edited files back to the starting point.

The idea is that you are an autonomous researcher trying one structural idea at a time. Keep a change only when quality goes up, or when quality stays the same and simplicity clearly goes up, and only when runtime does not increase. If it does not meet all of those conditions, throw it away and continue.

**Timeout**: The full `--max-cases 100` benchmark can take several minutes. If a run exceeds 15 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes due to a simple bug, fix it and re-run. If the idea itself is fundamentally broken, log `crash`, discard it, and move on.

**Priority order for ideas**: Focus on changes that remove special cases, reduce duplication, clarify invariants, simplify path extraction and pruning logic, and align the branch with `main`-style structure. Prefer structural cleanups and quality improvements over runtime-driven tricks.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The loop runs until the human interrupts you.
