# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `approx-2`.
3. **Read the in-scope files**: The repo is small enough to read the relevant approximation code for full context:
   - `README.md` — repository context.
   - `locomotif/locomotif.py` — motif-set loop and high-level orchestration.
   - `locomotif/loco.py` — LoCo wrapper, tau estimation, cumulative similarity entrypoints.
   - `locomotif/loco_jit.py` — core kernels, path extraction, sorting, pruning.
   - `/Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling/compact_profiling_all_quality.py` — benchmark harness. Do not modify.
4. **Verify the profiling workspace exists**: Check that the sister directory `/Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling` exists and that `cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --max-cases 10` runs there. If not, tell the human what is missing.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is benchmarked on the profiling workspace. The primary benchmark is:

`cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --max-cases 10`

The current baseline on `approx-2` is:

```
average_quality_score=0.6174
total_seconds=194.174353
```

**What you CAN do:**
- Modify `locomotif/locomotif.py`, `locomotif/loco.py`, and `locomotif/loco_jit.py`.
- Take inspiration from other branches, but do not rely on them blindly. Use them as references, and prefer finding your own simple, benchmark-backed improvements.
- Change algorithms, data structures, pruning, path extraction, tau estimation, cumulative similarity computation, and approximation logic.
- Rework any part of the algorithm if it improves the benchmark, as long as the result still maintains the spirit of the LoCoMotif algorithm.

**What you CANNOT do:**
- Modify the profiling harness in `/Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling`.
- Install new packages or add dependencies.
- Hardcode per-dataset behavior or tune special-case hyperparameters for individual datasets.
- Disable warping support.
- Replace LoCoMotif with a fundamentally different method that no longer matches the broad semantics of the original algorithm.

**Algorithmic reference**: The implementation may change substantially, but it should still preserve the broad LoCoMotif semantics from the paper at `/Users/fre/Documents/University/2025-2026/thesis/papers/LoCoMotif_discovering_time-warped_motifs_in_time_s.pdf`. Use that paper as the reference for what must remain true at a high level, rather than trying to preserve exact code behavior.

**The goal is simple: improve speed and quality together on the full `--max-cases 10` benchmark.** Favor global strategies that make sense across datasets and time-series lengths.

**Quality-first criterion**: When speed and quality move in opposite directions, quality comes first. Keep a change only if it improves the overall tradeoff. A speedup with a noticeable quality drop is usually not worth it. A small quality gain with similar runtime is worth it. Equal quality with a meaningful speedup is also worth it.

**Simplicity criterion**: All else being equal, simpler is better. A speedup or quality improvement that removes complexity is especially valuable. Do not keep extra machinery unless the benchmark win is real.

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
commit	avg_quality	total_seconds	cricket10_quality	cricket10_seconds	mallat10_quality	mallat10_seconds	status	description
```

1. git commit hash (short, 7 chars) or `wip`
2. average quality on the full `--max-cases 10` benchmark
3. total runtime on the full `--max-cases 10` benchmark
4. quality on `--benchmark cricket --max-cases 10`
5. runtime on `--benchmark cricket --max-cases 10`
6. quality on `--benchmark mallat --max-cases 10`
7. runtime on `--benchmark mallat --max-cases 10`
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:

```
commit	avg_quality	total_seconds	cricket10_quality	cricket10_seconds	mallat10_quality	mallat10_seconds	status	description
a1b2c3d	0.6174	194.17	0.8136	138.14	0.7546	31.03	keep	baseline
b2c3d4e	0.6210	186.50	0.8200	129.00	0.7600	28.80	keep	use smarter tau estimate
c3d4e5f	0.6120	175.10	0.7900	120.00	0.7400	26.00	discard	aggressive endpoint pruning
d4e5f6g	0.0000	0.00	0.0000	0.00	0.0000	0.00	crash	broken sparse backtracking
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar21`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune the approximation code with one experimental idea by directly hacking the code.
3. Run the screening benchmarks:
   - `cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --benchmark cricket --max-cases 10 > run_cricket.log 2>&1`
   - `cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --benchmark mallat --max-cases 10 > run_mallat.log 2>&1`
4. Read out the screening results:
   - `grep "^average_quality_score=\|^total_seconds=" run_cricket.log`
   - `grep "^average_quality_score=\|^total_seconds=" run_mallat.log`
5. If either screening run crashes, run `tail -n 50` on the corresponding log, decide whether the issue is fixable, and either retry or discard.
6. If the screening tradeoff is clearly bad, discard immediately and revert to where you started.
7. If screening looks promising, run the promotion benchmark:
   - `cd /Users/fre/Documents/University/2025-2026/thesis/code/locomotif-profiling && uv run python compact_profiling_all_quality.py --max-cases 10 > run.log 2>&1`
8. Read out the promotion results:
   - `grep "^average_quality_score=\|^total_seconds=" run.log`
9. Record the results in the TSV.
10. Promotion rule:
   - If full-run average quality improves and total runtime does not increase by more than 2%, keep.
   - If full-run average quality is unchanged within `0.002` and runtime improves by at least 5%, keep.
   - If full-run average quality improves by at least `0.005` and runtime also improves, keep.
   - Otherwise discard.
11. If the change is kept, git commit it and advance the branch.
12. If the change is discarded, revert only the edited files back to the starting point.

The idea is that you are an autonomous researcher trying one structural idea at a time. If it works, keep it. If it does not, throw it away and continue.

**Timeout**: The full `--max-cases 10` benchmark can take several minutes. If a run exceeds 15 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes due to a simple bug, fix it and re-run. If the idea itself is fundamentally broken, log `crash`, discard it, and move on.

**Priority order for ideas**: Prefer low-risk ports from `approx` first:
1. smarter tau estimation
2. block-pruned cumulative similarity
3. faster sorting/backpointer path extraction
4. only then larger approximation/data-structure changes

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The loop runs until the human interrupts you.
