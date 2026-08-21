# Domain-document layout

SparkCache uses a single-context domain-document layout.

- `CONTEXT.md`, when present, defines the repository-wide domain model,
  terminology, invariants, and module responsibilities.
- `docs/adr/`, when present, contains architectural decision records.
- `AGENTS.md` and `CLAUDE.md` define repository-wide rules for human and agent
  contributors.
- subsystem README files define interfaces, supported configurations, and
  qualification evidence for their directories.

Before changing architecture or terminology, read the applicable documents in
that order. If `CONTEXT.md` or `docs/adr/` does not exist, proceed using the
repository rules and subsystem documentation; absence does not imply an
undocumented decision.

Do not create a domain document solely to record development chronology.
Canonical domain documents specify present behavior, boundaries, and durable
decisions.
