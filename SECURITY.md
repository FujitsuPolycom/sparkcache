# Security policy

## Supported code

Security fixes target the repository's default branch. Published package
versions do not have a guaranteed maintenance window.

A report must name every affected version known to the reporter. The
maintainer may backport a fix when doing so is safe.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose prompt-derived
KV state, escape configured cache roots, bypass integrity checks, execute code,
or disclose credentials.

Use the repository's **Security** tab to open a private vulnerability report.
Include the affected revision, deployment conditions, impact, and a minimal
reproduction when it is safe to provide one.

If private reporting is unavailable, contact the repository owner privately.
Disclose only enough information to establish a secure reporting channel.

The maintainer will acknowledge a report when available and will coordinate a
fix and disclosure schedule appropriate to the impact. This personal project
does not provide a response-time or remediation-time service-level agreement.

## Deployment considerations

SparkCache stores prompt-derived KV tensors rather than raw prompt text. KV
state is still sensitive derived data.

Protect cache roots with appropriate filesystem permissions. Use separate
roots when tenants are not allowed to share derived state.

SparkCache verifies cache identity and stored data before restore. An identity,
integrity, or all-rank failure becomes a cache miss and recomputation.

These checks do not replace host hardening, storage encryption, access control,
or isolation of the model-serving runtime.
