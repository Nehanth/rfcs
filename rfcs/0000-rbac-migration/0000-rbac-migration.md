---
start_date: 2026-05-08
mlflow_issue:
rfc_pr:
---

<!-- markdownlint-disable-file MD041 -->

| Author(s)              | [Pat Sukprasert](https://github.com/PattaraS) |
| :--------------------- | :-------------------------------------------- |
| **Date Last Modified** | 2026-05-08                                    |

<!-- markdownlint-disable MD025 -->

# Summary: Role-Based Access Control for MLflow OSS

MLflow OSS today expresses authorization as per-user, per-resource permission
rows scattered across seven tables (one per resource type). There are no
roles, no groups, and no shared permission sets. Operators who want to grant
the same permission to multiple users do so one resource and one user at a
time. This RFC proposes replacing that model with a small role-based core:
**roles** carry permission rows, **users** are assigned to roles, and the
seven legacy tables collapse into a single `role_permissions` table.

The migration ships in two phases. Phase 1 added the role surface as a
parallel write path. Phase 2 (in progress) backfills the legacy tables into
`role_permissions`, deprecates the per-resource REST endpoints (with
deprecation warnings, targeted for removal one release later), removes the
workspace-permission endpoints, and stages a graduation migration that drops
the legacy tables outright. The end state is a single permission table and a
single grant API.

# Basic example

The shift in the user-facing flow is the main reason for this proposal, so we
ground the design in concrete scenarios that an operator faces today.

The admin UI surfaces two top-level entry points, both as modals with
pre-filled current state and a diff preview before submit:

- **Edit role** (from the Role detail page): sections for *Role details*,
  *Permissions*, *Assigned users*. Same shape as **Create role**.
- **Edit access** (from the User detail page): sections for *Role
  assignments*, *Direct permissions*, *Admin status*.

There is no separate "Assign user" or "Grant permission" button — both flows
live behind the unified modals.

### 1. Give Alice EDIT on experiment 42 (one-off)

**Today.** A single per-resource POST:

```http
POST /api/2.0/mlflow/experiments/permissions/create
{ "experiment_id": "42", "username": "alice", "permission": "EDIT" }
```

**Proposed.** Two paths, depending on whether the grant is genuinely one-off
or part of a recurring pattern.

*Direct permission* — closest 1-to-1 with the old call. Internally backed by
a synthetic per-user role (`__user_<id>__`); same `role_permissions`
storage, no special-casing in the resolver. UI: `/admin` → Users → Alice
→ **Edit access** → *Direct permissions* section → add `experiment:42 →
EDIT` → review → apply.

*Role-based* — better when the same grant will be reused.

```http
POST /api/3.0/mlflow/roles/create
  { "name": "exp-42-editor", "workspace": "ml-research" }
POST /api/3.0/mlflow/roles/permissions/add
  { "role_id": <id>, "resource_type": "experiment",
    "resource_pattern": "42", "permission": "EDIT" }
POST /api/3.0/mlflow/roles/assign
  { "username": "alice", "role_id": <id> }
```

UI: `/admin` → Roles → **Create role** → in the *Permissions* section add
`experiment:42 → EDIT`; in the *Assigned users* section add Alice → Create.

### 2. Give a team READ on every experiment in a workspace

**Today.** No good answer. Either set the workspace-level permission to
`READ` (a grab-bag that also covers registered models, gateway endpoints,
and scorers) or list every experiment and POST a permission row per (user ×
experiment), then redo it whenever a new experiment is created.

**Proposed.** A single role with a wildcard pattern. New experiments
inherit:

```http
POST /api/3.0/mlflow/roles/create
  { "name": "experiment-reader", "workspace": "ml-research" }
POST /api/3.0/mlflow/roles/permissions/add
  { "role_id": <id>, "resource_type": "experiment",
    "resource_pattern": "*", "permission": "READ" }
```

Then `roles/assign` for each team member (or add them all at once via the
modal). UI: Roles → **Create role** → `experiment` / `*` ("All experiments")
/ `READ`; *Assigned users* section → add the team → Create.

### 3. Make Alice a workspace admin

**Today.** A direct workspace-permission grant:

```http
POST /api/2.0/mlflow/workspaces/ml-research/permissions
{ "username": "alice", "permission": "MANAGE" }
```

The behavior of `MANAGE` here is opaque: depending on `default_permission`
configuration, it silently extended to every resource type.

**Proposed.** A default `admin` role is seeded on workspace creation. The
operator just assigns it:

```http
POST /api/3.0/mlflow/roles/assign
{ "username": "alice", "role_id": <admin-role-id> }
```

The role explicitly carries `(workspace, *, MANAGE)` — the same authority,
but the grant is named, listable, bulk-assignable, and that's the only thing
it confers. There is no implicit fan-out across resource types.

UI: Users → Alice → **Edit access** → *Role assignments* section → add
`ml-research/admin` → review → apply.

## Motivation

The pre-RBAC permission model has three structural problems.

First, there is no abstraction for "the same permission, applied to many
people." Every grant is a per-user-per-resource POST. Five users editing ten
experiments is fifty calls and fifty rows. Adding an eleventh experiment
requires fifty more calls. There is no built-in way to express "the
ml-research data scientists" as a unit.

Second, the storage is split across seven tables (`experiment_permissions`,
`registered_model_permissions`, the four `gateway_*_permissions` tables,
`scorer_permissions`, and `workspace_permissions`) with inconsistent
workspace-awareness. Some tables have a workspace column and some don't.
Resolving "what can Alice do here?" requires walking each table.

Third, the permission resolution logic carries cumulative complexity from
the per-table layout. Auth-server code that filters list / search results,
cascades on resource delete, and renames on rename has to fan out across all
seven shapes. The on-disk surface (~1000 LoC of per-resource CRUD, ~16 REST
endpoints, ~30 `AuthServiceClient` methods) is a function of that fan-out.

Roles fix all three. A role is a named, reusable bundle of permissions.
Storage collapses to one `role_permissions` table indexed by role. Resolution
is "look up the user's roles, union the permissions, evaluate against the
request" — no per-table fan-out.

### Out of scope

- **Teams / groups.** Roles can be reused across users, but there is no
  group abstraction yet. We expect this to come later if demand emerges; the
  role surface is forward-compatible with adding `group_role_assignments`.
- **Per-user / per-team budget policies.** Global gateway budgets exist;
  per-identity budgets are tracked separately and don't depend on this
  migration.
- **Cross-workspace roles.** Roles are workspace-scoped. A user who needs
  the same set of permissions in two workspaces is assigned to two
  workspace-scoped roles. We considered global roles but it complicates
  authorization without obvious benefit at this scale.
- **A "permission viewer" UI for end users.** The current admin UI is for
  operators. End users don't yet have a "what can I see?" view; would be a
  follow-up if requested.

## Detailed design

### Storage

```text
users  ─assigned─→  user_role_assignments  ─→  roles  ─have─→  role_permissions
                                                                      │
                                                                      └─→ (resource_type, resource_pattern, permission)
```

Three tables replace seven:

- `roles` — `(id, name, workspace, description)`. Workspace-scoped; a role
  named `editor` in workspace `foo` is a different row from `editor` in
  workspace `bar`.
- `role_permissions` — `(id, role_id, resource_type, resource_pattern,
  permission)`. The pattern is either a literal resource id or `*`
  (workspace-wide for that resource type).
- `user_role_assignments` — `(id, user_id, role_id)`. A user can hold any
  number of roles in any number of workspaces.

The legacy tables remain on disk through Phase 2 for rollback safety. They
are no longer read or written by the auth server post-backfill, but the
schema stays. The graduation PR (#23089) drops them; it is staged but
deferred until at least one full release cycle of bake time on the
simplified model.

### Per-user direct grants

Pre-RBAC, an operator could POST a per-resource permission for a single
user. We preserve that affordance by treating it as a synthetic per-user
role: each user has at most one `__user_<id>__` role per workspace, owned by
the auth system, surfaced in the admin UI's *Direct permissions* section
but otherwise invisible. Direct permissions are stored as
`role_permissions` rows under that synthetic role — no special-casing in the
resolver.

This means the same permission resolution path handles both "permission via
shared role" and "permission via direct grant." The difference is purely
authorial: a direct grant is one user's name on a permission, a role is a
named bundle that anyone can be assigned to.

### Workspace-level grants

The pre-RBAC `workspace_permissions` table held `READ` / `EDIT` / `MANAGE`
levels with implicit fan-out to every resource type. The new model collapses
this to a single workspace permission slot:
`(resource_type='workspace', resource_pattern='*', permission)` where
`permission` is `USE` (member: read everything in the workspace, create
experiments / registered models) or `MANAGE` (admin: full authority,
including role/user management within the workspace). No third tier — the
two-level model matched the actual usage we observed in the legacy tier.

### Permission resolution

The resolver runs on every protected request:

1. **Admin bypass.** If `is_admin = true`, short-circuit to allow.
2. **Role-derived grant on the resource.** Look up the user's roles in the
   request's workspace; union their permissions; check for a row with
   `(resource_type, resource_id)` that grants the requested permission.
   Specific patterns win over `*`.
3. **Workspace-level grant.** If no resource-specific row matched, check for
   a `(workspace, *, …)` row that satisfies the requested permission.
4. **Server `default_permission`.** Configured fallback.

This is the same chain the legacy resolver used; the table topology is
different but the precedence rules are unchanged.

### Identity model

Three identity tiers are enforceable at the validator layer.

A **super admin** has `is_admin = true` on the user row. Super admins
short-circuit all permission checks in `_before_request` — they are the sole
bearer of system-wide operations such as deleting users and bulk operations.

A **workspace admin** holds `(workspace, *, MANAGE)` (via any role) in one
or more workspaces. They can manage roles, users, and direct grants within
those workspaces but cannot perform system-wide operations. The auth-server
helper `list_workspace_admin_workspaces(user_id)` returns the set of
workspaces in which a user holds workspace-admin authority; the validator
layer composes this with the request's workspace to authorize role and user
edits.

A **regular user** is any other authenticated identity. Their authorization
flows entirely through role-derived permissions.

### Workspace context

Workspaces are conveyed through the URL via the `?workspace=<name>` query
param. A frontend component `WorkspaceRouterSync` extracts the param on
every navigation and sets the global `activeWorkspace` state, which most
hooks and outgoing-link helpers consult. Routes are partitioned into
*workspace-aware* (the param is auto-injected) and *global* (the param is
stripped); `/admin` is a hybrid: `/admin` itself is global (cross-workspace
platform-admin view), and `/admin/ws?workspace=…` is the per-workspace
management view.

### Filter-and-search behavior

The list / search endpoints (`search_experiments`, `search_logged_models`,
`search_registered_models`, `search_model_versions`, plus the GraphQL
model-version filter) need to apply role-derived authorization to results.
Pre-RBAC, each path had its own inline filter logic that walked the
appropriate per-resource permissions table. We collapse them onto a shared
helper, `_role_based_read_predicate(username, resource_type)`, which builds
a read-permission callable from `list_role_grants_for_user_in_workspace`
once per request. The handler then filters its result set with the
predicate.

### Wire surface

Phase 2 changes the wire surface in two parts.

**Deprecated** (still available; emit a deprecation warning; backed by
synthetic per-user role grants under the hood):

- `POST/GET/PATCH/DELETE` on `/mlflow/{experiments,registered-models,scorers}/permissions`
- `POST/GET/PATCH/DELETE` on `/mlflow/gateway/{secrets,endpoints,model-definitions}/permissions`
- The ~26 corresponding `AuthServiceClient` per-resource methods
  (`create_experiment_permission`, `update_registered_model_permission`, etc.)

These keep working for one release as a compatibility courtesy, then are
removed in a follow-up release. Existing scripts won't break on upgrade; new
code should use the role API directly.

**Removed** (calls 404):

- `POST/GET/PATCH/DELETE` on `/mlflow/workspaces/<workspace>/permissions`
- The four `*_workspace_permission` `AuthServiceClient` methods

The role surface (`create_role`, `add_role_permission`, `assign_role`,
`unassign_role`, `list_user_roles`, etc.) is the only path going forward.
Direct per-user grants land via the same role API: clients construct a
synthetic per-user role on demand, or — more commonly — go through the
admin UI's *Direct permissions* section.

## Drawbacks

**Breaking wire change.** The workspace-permission endpoints are removed
outright; the per-resource permission endpoints are deprecated for one
release and then removed. Any client that called `POST /experiments/permissions/create`
or a sibling needs to be rewritten to use the role API by the removal
release; calls to `POST /workspaces/<ws>/permissions` and the four
`*_workspace_permission` client methods break immediately. The migration is
one-way: we are not maintaining the old endpoints behind a flag past the
deprecation window.

**Two grant shapes for one logical operation.** "Grant Alice EDIT on
experiment 42" can land as a direct permission or as a role. The two have
the same effect at the resolver but different operational semantics — a
role is reusable and listable; a direct permission is convenient and
one-off. Operators have to make a choice each time. We mitigate this with
documentation: prefer roles, use direct permissions when the grant is
genuinely one user.

**Synthetic-role artifacts in role listings.** The `__user_<id>__` synthetic
roles exist as `roles` rows. The admin UI filters them out (the user-facing
"roles" view never shows them), but a low-level operator inspecting the
database will see them. We treat this as acceptable: it keeps the resolver
single-path, which is worth more than a hidden row.

**Migration backfill complexity.** The Alembic migration has to translate
seven tables' worth of grants into role rows correctly. The mapping is not
mechanical: workspace `READ` rewrites to `(workspace, *, USE)`; workspace
`EDIT` fans out to `(workspace, *, USE)` plus a type-wildcard `EDIT` on
every concrete resource type; scorer names need URL-encoding to match the
new pattern format. The migration is the load-bearing piece of the
backfill. We extensively tested it but the worst-case incorrect mapping
would be a silent over- or under-grant that survives until someone notices.

**Per-workspace UX double-bind.** The admin URL splits cleanly between
`/admin` (platform admin) and `/admin/ws?workspace=…` (per-workspace
management), but two URL shapes mean two breadcrumb-back targets. Our
detail pages currently land on `/admin` regardless of where the user came
from; that's a known cosmetic gap.

**No flag-controlled rollout.** We considered a feature flag that would
allow the auth server to read from either the legacy tables or
`role_permissions`. We rejected it because the cost of maintaining two
read paths exceeds the benefit, and because the simplified model is the
target end state regardless. Operators who need rollback fall back to the
retained legacy tables and a downgraded server build.

# Alternatives

**Keep per-resource permissions, add roles as a new layer.** Roles would
become bundles of "users to grant per-resource permissions to," not the
storage primitive. This was the natural minimal change but doesn't address
the fundamental cost of the per-resource model — it just adds another layer
on top.

**Per-table role tables.** A `experiment_roles`, `registered_model_roles`,
etc. set of tables, mirroring the legacy structure. We rejected this
because it preserves the per-table fan-out we want to eliminate. The whole
point is to consolidate.

**Path-based workspace routing.** `/admin/workspace/:name` instead of the
hybrid `/admin/ws?workspace=…`. Cleaner URLs, but `WorkspaceRouterSync` and
`prefixRouteWithWorkspace` are built around the query param and would need
extension. We chose the hybrid as the smaller change. A future RFC may
revisit this if a broader workspace-routing refactor is undertaken.

**Group / team primitive.** Teams as first-class identities, with role
assignments at the team level. Out of scope for this proposal but
forward-compatible with the role surface.

# Adoption strategy

The change is breaking, so adoption is staged.

**Phase 1 (landed)** added the role tables, role API, and role-based
permission resolver as a parallel surface. The legacy tables remained the
source of truth for permission decisions; roles existed but were not yet
load-bearing. This is a non-breaking addition.

**Phase 2 (in progress)** is the breaking part. The backfill migration
walks the seven legacy tables and writes equivalent rows into
`role_permissions` under synthetic per-user roles. The auth server flips to
reading from `role_permissions` only. The workspace-permission endpoints
are removed; the per-resource permission endpoints are deprecated (still
work, emit a deprecation warning) and slated for removal one release later.
The legacy tables remain on disk.

Operators upgrading need to:

1. **Audit clients.** Any deployment that relies on the per-resource
   permission endpoints needs to migrate to the role API. The synthetic
   per-user role pattern (`__user_<id>__`) is the closest direct analogue
   if a deployment really wants to keep per-user direct grants. A migration
   guide in the changelog will spell this out.
2. **Re-evaluate workspace permissions.** Workspace `MANAGE` semantics no
   longer fan out implicitly to every resource type. Operators who relied
   on the implicit fan-out need to either grant explicit `(resource_type,
   *, …)` rows under a workspace role, or use the seeded `admin` role
   (which carries `(workspace, *, MANAGE)`).
3. **Validate the backfill.** The Alembic migration runs at upgrade time.
   Operators with non-trivial permission setups should validate that
   role-derived grants match expectations on a staging copy before
   upgrading production. The migration is reversible by downgrade if
   needed.

**Phase 3 (deferred to MLflow 3.X+2)** is the graduation: drop the legacy
tables. By that point the simplified model has shipped in a release and had
at least one minor cycle of bake time. Until then the rollback path is
"downgrade the server."

**Owner-delegation behavior change.** Pre-RBAC, a resource creator with
`MANAGE` could delegate permissions on that resource to others. Post-RBAC,
grant delegation requires workspace-admin or super-admin authority.
Deployments that relied on owner-delegation either need to elevate
workspace admins or accept that rebinding goes through the role API. We
flag this in the changelog as a behavioral shift, not a bug.

# Open questions

**Graduation cadence.** The graduation PR drops the legacy tables. We've
proposed waiting one release cycle (MLflow 3.X+2 where 3.X is the release
that ships the backfill). Is one release enough bake time, or should we
hold longer to give operators time to validate? Open to feedback.

**Group / team support.** We deferred groups out of scope, but the
`role_permissions` storage shape is forward-compatible with adding
`group_role_assignments`. Should we sketch the group surface in this RFC
to lock in the upgrade path, or leave it for a future RFC when the demand
crystallizes?

**Direct-permission ergonomics.** The synthetic per-user role pattern is
clean at the storage layer but slightly awkward to explain. An alternative
would be a separate `direct_permissions` table (one row per direct grant),
keeping `role_permissions` strictly for shared roles. This would simplify
the mental model at the cost of two read paths in the resolver. We chose
the synthetic-role path; flagging in case reviewers feel the trade-off
should go the other way.

**Per-workspace admin UI structure.** The current implementation splits
`/admin` (platform) from `/admin/ws?workspace=…` (per-workspace). A
broader per-workspace UX (every workspace gets its own admin entry point,
no global view) was considered and deferred. We treat it as a follow-up;
no consensus needed in this RFC.
