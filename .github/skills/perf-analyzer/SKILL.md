---
name: perf-analyzer
description: "Use when analyzing performance hotspots, complexity costs, cache behavior, vectorization opportunities, and benchmarking strategy."
---

# Perf Analyzer Skill

## When to use
- Analyze hot paths and scaling behavior.
- Identify likely bottlenecks before optimization.
- Propose measurement-first optimization plans.

## Inputs expected
- Function, file, or benchmark target.
- Optional workload shape and latency goals.

## Workflow
1. Inspect implementation and call paths.
2. Estimate algorithmic and runtime costs.
3. Recommend targeted measurements and improvements.

## Output format
- Performance findings
- Expected impact
- Measurement or optimization suggestions

## Guardrails
- Distinguish measured facts from hypotheses.
- Prioritize high-impact, low-risk changes.
