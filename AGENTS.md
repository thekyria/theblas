# AGENTS.md

This repository exposes custom Copilot agents from `.github/agents`.

## Available Agents

- Explore: [.github/agents/Explore.agent.md](.github/agents/Explore.agent.md)
- CodeReviewer: [.github/agents/CodeReviewer.agent.md](.github/agents/CodeReviewer.agent.md)
- Tester: [.github/agents/Tester.agent.md](.github/agents/Tester.agent.md)
- DocWriter: [.github/agents/DocWriter.agent.md](.github/agents/DocWriter.agent.md)
- SecurityAuditor: [.github/agents/SecurityAuditor.agent.md](.github/agents/SecurityAuditor.agent.md)
- PerfAnalyzer: [.github/agents/PerfAnalyzer.agent.md](.github/agents/PerfAnalyzer.agent.md)

## Notes

- Keep this file as a lightweight index.
- Put agent behavior, tool access, and usage guidance in the corresponding `.agent.md` files.
- Skills live under `.github/skills/<skill-name>/SKILL.md` and remain separate from custom agents.
- Skills complement agents for reusable workflows; they do not replace `.agent.md` definitions.

## Skills

- Explore: [.github/skills/explore/SKILL.md](.github/skills/explore/SKILL.md)
- Code Reviewer: [.github/skills/code-reviewer/SKILL.md](.github/skills/code-reviewer/SKILL.md)
- Tester: [.github/skills/tester/SKILL.md](.github/skills/tester/SKILL.md)
- Doc Writer: [.github/skills/doc-writer/SKILL.md](.github/skills/doc-writer/SKILL.md)
- Security Auditor: [.github/skills/security-auditor/SKILL.md](.github/skills/security-auditor/SKILL.md)
- Perf Analyzer: [.github/skills/perf-analyzer/SKILL.md](.github/skills/perf-analyzer/SKILL.md)
- Add Educational Comments: [.github/skills/add-educational-comments/SKILL.md](.github/skills/add-educational-comments/SKILL.md)
