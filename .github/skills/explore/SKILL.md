---
name: explore
description: "Use when exploring the repository, locating symbols, tracing references, or answering codebase questions with read-only analysis."
---

# Explore Skill

## When to use
- Find files, symbols, or call paths quickly.
- Answer architecture and ownership questions from source.
- Gather concrete references before coding.

## Inputs expected
- A natural-language question or search goal.
- Optional scope (folder, file, or symbol).
- Optional depth: quick, medium, or thorough.

## Workflow
1. Search for candidate files and symbols.
2. Read only the relevant source and docs.
3. Return evidence-backed findings with file references.

## Output format
- Objective
- Findings
- Relevant files
- Open questions

## Guardrails
- Do not edit files.
- Do not run write operations.
