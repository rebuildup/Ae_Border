# AGENTS.md — Ae_Border project dispatcher

## Governance

- Top-level contract: [`constitution/CONSTITUTION.md`](constitution/CONSTITUTION.md)
- Current Operating Model: [`organization/profiles/release-driven-solo.md`](organization/profiles/release-driven-solo.md)
- Project/build facts: `README.md`, source, project files, and accepted project docs

## Working rules

- Treat After Effects SDK, host version, Visual Studio/Xcode configuration, and generated plugin binaries as explicit environment/artifact identities.
- Do not infer current Adobe SDK behavior from memory when a version-sensitive change matters.
- Preserve unrelated SDK checkout and host state; do not use destructive cleanup outside repository-owned paths.
- Source-only checks do not prove host behavior. Record host verification separately against the tested candidate.
- Durable work/dependency state belongs in GitHub Issues when available; review/integration evidence belongs in Pull Requests.
