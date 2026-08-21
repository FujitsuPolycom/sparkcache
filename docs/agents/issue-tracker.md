# Local issue tracker

SparkCache uses local Markdown files for implementation plans, product
requirements, and issue records because this checkout has no configured Git
remote.

Store each work item under `.scratch/<feature-slug>/`. Use descriptive slugs
that identify the affected behavior, such as
`.scratch/bounded-nvme-eviction/`. A work-item directory may contain:

- `README.md` for the issue statement, scope, acceptance conditions, and
  status;
- `PRD.md` for product requirements;
- `plan.md` for an implementation plan;
- supporting evidence whose purpose and provenance are stated in the file.

Repository prose rules in `AGENTS.md` and `CLAUDE.md` apply to every work-item
file. Records must describe the desired or resulting behavior directly and
must not depend on conversation history.

If a Git remote and hosted issue tracker are configured, update this document
before creating hosted issues. Local Markdown remains authoritative until the
repository documents a different tracker.
