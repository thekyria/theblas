---
name: security-auditor
description: "Use when auditing code for vulnerabilities, unsafe memory handling, trust-boundary gaps, and practical hardening opportunities."
---

# Security Auditor Skill

## When to use
- Review sensitive code paths and input validation.
- Identify exploit paths and unsafe assumptions.
- Recommend concrete mitigations by impact.

## Inputs expected
- File, module, or data flow to audit.
- Optional threat concern (memory safety, injection, auth, secrets).

## Workflow
1. Inspect relevant code paths and boundaries.
2. Identify concrete vulnerabilities and preconditions.
3. Recommend prioritized mitigations.

## Output format
- Findings by severity
- Impact
- Recommended mitigations

## Guardrails
- Avoid generic advice not grounded in code.
- Clearly separate confirmed risk from suspicion.
