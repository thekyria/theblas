---
name: tester
description: "Use when running tests, reproducing failures, validating a fix, or summarizing test outcomes for a specific target."
---

# Tester Skill

## When to use
- Run unit or integration tests.
- Reproduce a failing test and isolate the root area.
- Confirm that changes pass targeted test commands.

## Inputs expected
- Test command, test pattern, or target module.
- Optional verbosity and failure focus.

## Workflow
1. Pick the narrowest command that answers the request.
2. Execute tests and capture output.
3. Summarize pass/fail status and failing components.

## Output format
- Command run
- Result summary
- Failures or warnings
- Suggested next step

## Guardrails
- Keep execution scoped to requested tests.
- Report failures accurately with key output details.
