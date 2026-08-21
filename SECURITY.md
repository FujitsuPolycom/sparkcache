# Security policy

## Supported code

Security fixes target the repository's default branch. Published package
versions do not have a guaranteed maintenance window; a report must identify
every exact version known to be affected. The maintainer may publish a patch
for an affected version when the fix can be backported safely.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose prompt-derived
KV state, escape configured cache roots, bypass integrity checks, execute code,
or disclose credentials.

Use the repository's **Security** tab to open a private vulnerability report.
Include the affected revision or package version, deployment conditions, impact,
and a minimal reproduction when it is safe to provide one. If private reporting
is not enabled on the repository host, contact the repository owner privately
and disclose only enough information to establish a secure reporting channel.

The maintainer will acknowledge a report when available and will coordinate a
fix and disclosure schedule appropriate to the impact. This personal project
does not provide a response-time or remediation-time service-level agreement.

## Deployment considerations

SparkCache stores prompt-derived KV tensors rather than raw prompt text. KV
state is still sensitive derived data. Operators must protect cache roots with
filesystem permissions appropriate to the model service and must use separate
cache roots when tenants are not permitted to share derived state.

SparkCache verifies cache identity and stored data before restore. An identity,
integrity, or quorum failure must degrade to recomputation. This fail-closed
behavior does not replace host hardening, storage encryption, access control,
or isolation of the model-serving runtime.
