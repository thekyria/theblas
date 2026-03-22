# AGENTS.md

This repository contains an AI assistant (Copilot) customization setup.

## Purpose

Document available agents and their roles in this workspace.

## Agents

### Explore
- Purpose: Fast read-only codebase exploration and Q&A.
- Usage: Search for code patterns, identify symbols, and gather context quickly.
- Notes: Used internally by the workflow to avoid manual search/read loops.

### CodeReviewer
- Purpose: Static audit and lint summary.
- Usage: `CodeReviewer: inspect src/ and report style/bug issues`
- Notes: Non-destructive, no code write.

### Tester
- Purpose: Execute test suites and summarize results.
- Usage: `Tester: run ctest -R test_theblas` or `Tester: run tests for module xyz`.
- Notes: Report pass/fail counts and failure details.

### DocWriter
- Purpose: Generate or improve documentation for code and workflows.
- Usage: `DocWriter: write README section for build instructions`.
- Notes: Non-invasive; produce markdown or structured docs.

### SecurityAuditor
- Purpose: Detect security issues and suggest hardening improvements.
- Usage: `SecurityAuditor: scan code for buffer overflows, unsafe casts, and missing validation`.
- Notes: Focus on analysis and mitigation recommendations.

### PerfAnalyzer
- Purpose: Analyze performance hotspots and suggest optimization strategies.
- Usage: `PerfAnalyzer: examine matrix multiply loops for cache and vectorization issues`.
- Notes: Identify changes with clear expected impact.

## How to use

1. For read-only searches and code discovery, use `Explore`.
2. For static analysis and style feedback, use `CodeReviewer`.
3. For running tests and summarizing results, use `Tester`.
4. For writing or updating documentation, use `DocWriter`.
5. For security auditing and mitigation recommendations, use `SecurityAuditor`.
6. For performance analysis and optimization suggestions, use `PerfAnalyzer`.

## Maintenance

- Keep this file lightweight and updated when new agents are added.
- For custom agent declarations, include:
  - Name
  - Description
  - Input/Output contract (if any)
  - Additional notes / constraints
