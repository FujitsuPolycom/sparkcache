# Domain-document layout

SparkCache uses a single-context domain-document layout.

- `CONTEXT.md`, when present, defines the repository-wide domain model,
  terminology, invariants, and module responsibilities.
- `docs/adr/`, when present, contains architectural decision records.
- `AGENTS.md` and `CLAUDE.md` define repository-wide rules for human and agent
  contributors.
- subsystem README files define interfaces and supported configurations;
- deployment profiles contain model-specific launch details and live test
  records.

Before changing architecture or terminology, read the applicable documents in
that order.

If `CONTEXT.md` or `docs/adr/` does not exist, use the repository rules and
subsystem documentation. Absence does not imply an undocumented decision.

Do not create a domain document solely to record development chronology.
Canonical domain documents specify present behavior, boundaries, and durable
decisions.
