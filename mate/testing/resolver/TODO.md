# Resolver TODO

This file records unresolved design and implementation work. It is not a source
dependency manifest. The operator catalog must continue to declare only the
operator name, top-level interfaces, and pytest targets.

## Performance

### Add snapshot-scoped caches

Selection now reuses one dependency resolver per unique snapshot, caches final
operation results for the request, and reuses one result with both provenance
labels when target and head identify the same commit. Lower-level Python parse
and symbol-closure caching remains incomplete.

Implement request-scoped caching first:

- Share parsed Python modules, JIT analyses, and lower-level symbol closures
  across operators in that snapshot.

Consider persistent caching only after measuring the request-scoped design.
Persistent keys should use Git tree/blob identities, a catalog fingerprint, and
an explicit resolver schema version. Dirty worktree results should initially
remain process-local; timestamp-only invalidation is not sufficient.
