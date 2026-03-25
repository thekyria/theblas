---
name: code-reviewer
description: "Use when reviewing code for bugs, regressions, risky behavior, missing tests, or static analysis concerns without editing code."
---

# Code Reviewer Skill

## When to use
- Review a file, module, or diff for correctness risks.
- Identify regression and reliability issues.
- Check for missing test coverage.

## Inputs expected
- File path, module, or change area.
- Optional risk focus (correctness, maintainability, tests, security).

## Workflow
1. Inspect code and related context.
2. Prioritize findings by severity.
3. Explain impact and recommended fixes.

## Output format
- Findings by severity
- Open questions or assumptions
- Optional summary

## Guardrails
- Do not edit files.
- Focus on actionable, code-grounded findings.
