# Update Plan Data Model

The update plan describes which observations are allowed to update which
parameters, which MDA algorithm the plan uses, and for ES-MDA which localization
behavior applies to each allowed relationship.

The central invariant is:

```text
Within one MDA assimilation step, one parameter atom may assimilate
one observation datum at most once.
```

This is the guardrail that prevents double-counting the same likelihood
contribution. The same observation may update different parameter atoms. The
same parameter atom may be updated by multiple observation groups only when the
resolved observation sets are disjoint. Those disjoint observations are then
assimilated by the plan's single MDA algorithm, not by competing
relationship-level algorithms.

## Modeling Goals

- Support all parameter types: scalar parameters, fields, and surfaces.
- Support user-configurable relationships between parameter groups and
  observation groups without requiring a full Cartesian product of groups.
- Store relationship-specific behavior on the parameter-group/observation-group
  relationship, not on observations or parameters.
- Allow a UI to present a matrix or table for small or filtered subsets, but do
  not make a dense matrix the authoritative configuration model.
- Choose the MDA algorithm at the update-plan level: `ES_MDA` or `ENIF_MDA`.
- For `ES_MDA`, allow each enabled relationship to choose localization behavior,
  such as adaptive localization for scalars and distance-based localization for
  fields and surfaces.
- Prevent accidental overlap where the same parameter atom is updated by the
  same observation through multiple enabled relationships.
- Keep relational tables small and metadata-oriented.
- Store large parameter values, masks, responses, and resolved coverage sets in
  object storage as Parquet, Arrow, Zarr, NetCDF, or another explicit format.
- Make resolution and validation reproducible through content hashes and
  resolved artifacts.

## Core Concepts

### Parameter Atom

A parameter atom is the smallest coordinate in the parameter state vector
that can be updated by the assimilation algorithm. It is an identity, not a
single stored value.

| Parameter type | Parameter atom |
| --- | --- |
| `SCALAR` | One scalar key or control, such as one `GEN_KW` entry |
| `FIELD` | One active field cell/value in a named field geometry |
| `SURFACE` | One active surface node/cell in a named surface geometry |

The atom identity must be stable across realizations in the ensemble. The
ensemble values for one atom are the samples of that coordinate across active
realizations.

For fields and surfaces, a parameter atom is typically identified by
`(parameter_id, atom_index)`, where `atom_index` is defined by a geometry or
atom-index asset. It may correspond to `(i, j, k)` for a field or `(i, j)` or a
mesh node id for a surface. The atom does not need to be stored as a relational
row.

For scalars, parameter atoms are small enough to store directly as rows in
`scalar_parameter_atom`.

### Parameter Asset

A configured ERT parameter object. A parameter has one of three types:

| Type | Examples | Parameter atom |
| --- | --- | --- |
| `SCALAR` | `GEN_KW`, controls, scalar coefficients | One scalar key/control |
| `FIELD` | 3D grid property such as `PORO` | One active grid cell/value |
| `SURFACE` | 2D map/surface | One active surface node/cell |

For fields and surfaces, the parameter atoms can number in the millions. They
must not be stored as one relational row per cell unless the data set is small.
Instead, atom identity is defined by stable indices in a geometry or atom-index
asset.

### Observation Datum

One scalar observation value with a standard deviation and a stable identity,
normally `(observation_key, obs_index)` within an experiment. Observations may
also have location metadata used by localization methods.

### Parameter Group

A named selector over parameter atoms. A group may represent scalar
keys, a full field, a field region, a surface polygon, an active mask, or any
other supported parameter subset. A group should be homogeneous in
`parameter_type`.

Parameter groups should reduce configuration cardinality,
meaning that a group usually contains many parameter atoms,
such as a field region or a set of scalar keys.
If a workflow needs atom-level control, use selectors,
rules, and resolved artifacts instead of making users manage one group per atom.

### Observation Group

A named selector over observation datums. Observation groups may overlap. The
overlap becomes invalid only when overlapping observations are enabled for
overlapping parameter atoms in the same assimilation step.

Observation groups should also reduce cardinality. If a workflow needs one group
per observation datum or per small time/location bucket, relationship rules and
filtered views are preferable to a manually edited grid.

### Update Relationship

One sparse relationship between a parameter group and an observation group,
stored explicitly in `update_pair` when a relationship or override needs to be
recorded:

```text
parameter_group x observation_group -> enabled, localization_type, localization_settings
```

Missing relationships are disabled by default and produce no resolved update
coverage. Disabled `update_pair` rows are only needed when they record an
explicit deny or override a broader relationship rule. In an `ES_MDA` plan,
enabled relationships choose the localization behavior for that parameter-group
and observation-group relationship. In an `ENIF_MDA` plan, enabled relationships
only define which relationships are active; EnIF-MDA is the plan-level algorithm
and applies to all enabled relationships.

A UI may project update relationships into a matrix for small or filtered
subsets. The stored model should remain sparse so that plans with thousands of
parameters, observations, or groups are not forced to materialize every possible
cell.

### Update Relationship Rule

One rule that can enable, disable, or configure many parameter-group and
observation-group relationships. Rules are useful when many groups share the
same localization policy or when relationships are selected by name, type,
metadata, region, or another selector.

Rules are resolved into concrete enabled relationships before validation and
execution. If an explicit `update_pair` row and a rule both match the same
relationship, the explicit row should take precedence.

Rules should be the normal way to express broad policies. Explicit relationships
should be used for small plans, important exceptions, or user-reviewed overrides.

### Resolved Coverage

The concrete result of evaluating selectors, relationship rules, and explicit
update relationships against one experiment state. Resolved coverage is what
validation and execution consume.

For scalar groups, it is reasonable to materialize edges in relational tables.
For fields and surfaces, resolved parameter coverage should usually be stored as
mask assets in object storage, with only metadata and asset references in the
database.

## Storage Split

The relational database stores:

- Parameter, observation, group, relationship, and relationship-rule metadata.
- Selector rules.
- References to large artifacts.
- Content hashes, schema versions, counts, and provenance.
- Validation results and conflict reports.

Object storage stores:

- Parameter ensemble values.
- Field and surface active masks.
- Field and surface geometry or atom-index mappings.
- Large resolved parameter-group masks.
- Large resolved observation sets if needed.
- Response matrices and other bulk arrays.
- Optional localization weights or algorithm-specific precomputed artifacts.

Parquet is a good fit for tabular scalar data, sparse masks, observation sets,
and conflict samples. Zarr or NetCDF-like chunked arrays are usually better for
dense multidimensional field and surface values.

## Relational Tables

The SQL below is conceptual. Concrete implementations may use database-specific
enum types, JSONB, partitioning, or additional indexes.

### Data Assets

`data_asset` is the common handle for anything stored outside the relational
database.

```sql
CREATE TABLE data_asset (
    id                     BIGINT PRIMARY KEY,
    experiment_id          BIGINT NOT NULL,
    asset_type             TEXT NOT NULL,
    storage_uri            TEXT NOT NULL,
    storage_format         TEXT NOT NULL,
    schema_version         INTEGER NOT NULL,
    content_hash           TEXT NOT NULL,
    metadata               JSON NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, storage_uri)
);
```

Examples of `asset_type`:

- `PARAMETER_VALUES`
- `PARAMETER_ACTIVE_MASK`
- `PARAMETER_ATOM_INDEX`
- `PARAMETER_GEOMETRY`
- `OBSERVATION_SET`
- `RESPONSE_VALUES`
- `DESIGN_MATRIX`
- `SENSITIVITY_DESIGN`
- `ANALYSIS_RESULT`
- `SENSITIVITY_RESULT`
- `RESOLVED_PARAMETER_MASK`
- `RESOLVED_OBSERVATION_SET`
- `LOCALIZATION_WEIGHTS`
- `ENIF_GRAPH`
- `ENIF_PRECISION`

### Parameters

`parameter` stores the configured parameter object. It does not store the bulk
ensemble values.

```sql
CREATE TABLE parameter (
    id                     BIGINT PRIMARY KEY,
    experiment_id          BIGINT NOT NULL,
    name                   TEXT NOT NULL,
    parameter_type         TEXT NOT NULL CHECK (
                               parameter_type IN ('SCALAR', 'FIELD', 'SURFACE')
                           ),
    geometry_asset_id      BIGINT REFERENCES data_asset(id),
    definition_settings    JSON NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, name)
);

CREATE TABLE parameter_value_asset (
    id                     BIGINT PRIMARY KEY,
    parameter_id           BIGINT NOT NULL REFERENCES parameter(id),
    ensemble_id            BIGINT NOT NULL,
    values_asset_id        BIGINT NOT NULL REFERENCES data_asset(id),
    active_mask_asset_id   BIGINT REFERENCES data_asset(id),
    atom_index_asset_id    BIGINT REFERENCES data_asset(id),
    realization_count      INTEGER NOT NULL,
    atom_count             BIGINT NOT NULL,
    UNIQUE (parameter_id, ensemble_id)
);
```

`atom_index_asset_id` defines stable atom indices for fields and surfaces. For a
field this can map `atom_index` to `(i, j, k)` and optionally `(x, y, z)`. For a
surface this can map `atom_index` to `(i, j)` or a node id and optionally
`(x, y)`.

Scalar atoms are small enough to store directly.

```sql
CREATE TABLE scalar_parameter_atom (
    id                     BIGINT PRIMARY KEY,
    parameter_id           BIGINT NOT NULL REFERENCES parameter(id),
    atom_key               TEXT NOT NULL,
    atom_index             BIGINT NOT NULL,
    metadata               JSON NOT NULL DEFAULT '{}',
    UNIQUE (parameter_id, atom_key),
    UNIQUE (parameter_id, atom_index)
);
```

### Observations And Responses

Observation metadata can usually be stored directly in the relational database.
Bulk response matrices belong in object storage.

```sql
CREATE TABLE observation (
    id                     BIGINT PRIMARY KEY,
    experiment_id          BIGINT NOT NULL,
    observation_key        TEXT NOT NULL,
    obs_index              BIGINT NOT NULL,
    response_key           TEXT,
    value                  DOUBLE PRECISION NOT NULL,
    std                    DOUBLE PRECISION NOT NULL,
    east                   DOUBLE PRECISION,
    north                  DOUBLE PRECISION,
    tvd                    DOUBLE PRECISION,
    time_value             TIMESTAMP,
    default_radius         DOUBLE PRECISION,
    metadata               JSON NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, observation_key, obs_index)
);

CREATE TABLE response_value_asset (
    id                     BIGINT PRIMARY KEY,
    experiment_id          BIGINT NOT NULL,
    ensemble_id            BIGINT NOT NULL,
    response_type          TEXT NOT NULL,
    values_asset_id        BIGINT NOT NULL REFERENCES data_asset(id),
    observation_axis_hash  TEXT,
    realization_count      INTEGER NOT NULL,
    response_count         BIGINT NOT NULL,
    UNIQUE (experiment_id, ensemble_id, response_type)
);
```

### Update Plan And MDA Steps

An update plan may be applied to one or more MDA assimilation steps. The same
observations may be reused across different MDA steps with the proper step
scaling. The no-double-counting invariant applies within a single step.

```sql
CREATE TABLE update_plan (
    id                     BIGINT PRIMARY KEY,
    experiment_id          BIGINT NOT NULL,
    name                   TEXT NOT NULL,
    algorithm_type         TEXT NOT NULL CHECK (
                               algorithm_type IN ('ES_MDA', 'ENIF_MDA')
                           ),
    settings               JSON NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, name)
);

CREATE TABLE assimilation_step (
    id                     BIGINT PRIMARY KEY,
    plan_id                BIGINT NOT NULL REFERENCES update_plan(id),
    step_index             INTEGER NOT NULL,
    observation_scaling    DOUBLE PRECISION NOT NULL,
    settings               JSON NOT NULL DEFAULT '{}',
    UNIQUE (plan_id, step_index)
);

```

`algorithm_type` is a plan-level choice. Users define an experiment as either
`ENIF_MDA` or `ES_MDA`. EnIF-MDA is not mixed with ES-MDA localization choices
inside the same update plan.

### Parameter Groups

Parameter groups are resolved from ordered selectors. Later selectors override
earlier selectors for the atoms they match.

```sql
CREATE TABLE parameter_group (
    id                     BIGINT PRIMARY KEY,
    plan_id                BIGINT NOT NULL REFERENCES update_plan(id),
    name                   TEXT NOT NULL,
    parameter_type         TEXT NOT NULL CHECK (
                               parameter_type IN ('SCALAR', 'FIELD', 'SURFACE')
                           ),
    settings               JSON NOT NULL DEFAULT '{}',
    UNIQUE (plan_id, name)
);

CREATE TABLE parameter_group_selector (
    id                     BIGINT PRIMARY KEY,
    group_id               BIGINT NOT NULL REFERENCES parameter_group(id),
    rule_order             INTEGER NOT NULL,
    selector_type          TEXT NOT NULL,
    parameter_id           BIGINT REFERENCES parameter(id),
    enabled_override       BOOLEAN,
    selector_settings      JSON NOT NULL DEFAULT '{}',
    selector_asset_id      BIGINT REFERENCES data_asset(id),
    UNIQUE (group_id, rule_order)
);
```

Examples of `selector_type`:

- `ALL_PARAMETER_ATOMS`
- `PARAMETER_NAME_PATTERN`
- `SCALAR_KEY_PATTERN`
- `ATOM_INDEX_RANGE`
- `FIELD_REGION_MASK`
- `SURFACE_REGION_MASK`
- `POLYGON`
- `DEPTH_INTERVAL`
- `ASSET_MASK`

Resolved scalar membership can be stored directly.

```sql
CREATE TABLE resolved_scalar_parameter_group_member (
    group_id               BIGINT NOT NULL REFERENCES parameter_group(id),
    scalar_atom_id         BIGINT NOT NULL REFERENCES scalar_parameter_atom(id),
    PRIMARY KEY (group_id, scalar_atom_id)
);
```

Resolved field and surface membership should normally be stored as mask assets.

```sql
CREATE TABLE resolved_parameter_group_asset (
    group_id               BIGINT NOT NULL REFERENCES parameter_group(id),
    parameter_id           BIGINT NOT NULL REFERENCES parameter(id),
    atom_mask_asset_id     BIGINT NOT NULL REFERENCES data_asset(id),
    atom_count             BIGINT NOT NULL,
    membership_hash        TEXT NOT NULL,
    PRIMARY KEY (group_id, parameter_id)
);
```

The mask asset should identify selected atoms by stable `atom_index`, not by
physical value. It may be dense, sparse, chunked, or compressed.

### Observation Groups

Observation groups are also resolved from ordered selectors. Precedence is
local to one observation group.

```sql
CREATE TABLE observation_group (
    id                     BIGINT PRIMARY KEY,
    plan_id                BIGINT NOT NULL REFERENCES update_plan(id),
    name                   TEXT NOT NULL,
    settings               JSON NOT NULL DEFAULT '{}',
    UNIQUE (plan_id, name)
);

CREATE TABLE observation_group_rule (
    id                       BIGINT PRIMARY KEY,
    group_id                 BIGINT NOT NULL REFERENCES observation_group(id),
    rule_order               INTEGER NOT NULL,
    selector_type            TEXT NOT NULL,
    observation_key_pattern  TEXT,
    obs_index                BIGINT,
    enabled_override         BOOLEAN,
    radius_override          DOUBLE PRECISION,
    rule_settings            JSON NOT NULL DEFAULT '{}',
    selector_asset_id        BIGINT REFERENCES data_asset(id),
    UNIQUE (group_id, rule_order)
);
```

Examples of `selector_type`:

- `OBSERVATION_KEY_PATTERN`
- `OBSERVATION_INDEX`
- `RESPONSE_KEY_PATTERN`
- `TIME_RANGE`
- `SPATIAL_REGION`
- `ASSET_SET`

Resolved observation membership is usually small enough to store directly.

```sql
CREATE TABLE resolved_observation_group_member (
    group_id               BIGINT NOT NULL REFERENCES observation_group(id),
    observation_id         BIGINT NOT NULL REFERENCES observation(id),
    effective_radius       DOUBLE PRECISION,
    effective_settings     JSON NOT NULL DEFAULT '{}',
    PRIMARY KEY (group_id, observation_id)
);
```

If an observation group is too large for direct membership rows, use an asset
reference as the authoritative resolved set.

```sql
CREATE TABLE resolved_observation_group_asset (
    group_id               BIGINT PRIMARY KEY REFERENCES observation_group(id),
    observation_set_asset_id BIGINT NOT NULL REFERENCES data_asset(id),
    observation_count      BIGINT NOT NULL,
    membership_hash        TEXT NOT NULL
);
```

### Update Relationships

`update_pair` stores explicit relationships or overrides between one parameter
group and one observation group. It is intentionally sparse: absence of a row
means the relationship is disabled unless an `update_pair_rule` enables it.
Rows with `enabled = FALSE` are only needed to record explicit denies, such as
exceptions to a broad rule. A UI may display these relationships as a matrix for
small or filtered subsets, but a dense `parameter_group x observation_group`
matrix is not the storage model.

```sql
CREATE TABLE update_pair (
    id                     BIGINT PRIMARY KEY,
    plan_id                BIGINT NOT NULL REFERENCES update_plan(id),
    parameter_group_id     BIGINT NOT NULL REFERENCES parameter_group(id),
    observation_group_id   BIGINT NOT NULL REFERENCES observation_group(id),
    enabled                BOOLEAN NOT NULL DEFAULT FALSE,
    localization_type      TEXT CHECK (
                               localization_type IS NULL OR localization_type IN (
                                   'ADAPTIVE_LOCALIZATION',
                                   'NO_LOCALIZATION',
                                   'DISTANCE_LOCALIZATION'
                               )
                           ),
    localization_settings  JSON NOT NULL DEFAULT '{}',
    pair_settings          JSON NOT NULL DEFAULT '{}',
    UNIQUE (plan_id, parameter_group_id, observation_group_id)
);
```

`update_pair_rule` is optional. It provides a compact way to enable, disable, or
configure many relationships without materializing every group combination. Rule
settings may match parameter group names, observation group names, parameter
types, metadata, or other implementation-defined selectors. Later rules override
earlier matching rules for the columns they set; explicit `update_pair` rows
override rules.

```sql
CREATE TABLE update_pair_rule (
    id                          BIGINT PRIMARY KEY,
    plan_id                     BIGINT NOT NULL REFERENCES update_plan(id),
    rule_order                  INTEGER NOT NULL,
    parameter_group_pattern     TEXT,
    observation_group_pattern   TEXT,
    parameter_type              TEXT CHECK (
                                    parameter_type IS NULL OR parameter_type IN (
                                        'SCALAR',
                                        'FIELD',
                                        'SURFACE'
                                    )
                                ),
    enabled_override            BOOLEAN,
    localization_type           TEXT CHECK (
                                    localization_type IS NULL OR localization_type IN (
                                        'ADAPTIVE_LOCALIZATION',
                                        'NO_LOCALIZATION',
                                        'DISTANCE_LOCALIZATION'
                                    )
                                ),
    localization_settings       JSON NOT NULL DEFAULT '{}',
    rule_settings               JSON NOT NULL DEFAULT '{}',
    UNIQUE (plan_id, rule_order)
);
```

Implementations may replace the pattern columns with a more general selector
shape if needed. The important property is that rule resolution produces a
sparse set of enabled relationships and explicit denies, not a full Cartesian
product.

`resolved_update_relationship` records the effective relationship after applying
rules and explicit overrides. Validation and execution consume this table, not
the raw `update_pair` and `update_pair_rule` rows.

```sql
CREATE TABLE resolved_update_relationship (
    id                          BIGINT PRIMARY KEY,
    plan_id                     BIGINT NOT NULL REFERENCES update_plan(id),
    parameter_group_id          BIGINT NOT NULL REFERENCES parameter_group(id),
    observation_group_id        BIGINT NOT NULL REFERENCES observation_group(id),
    source_update_pair_id       BIGINT REFERENCES update_pair(id),
    source_rule_ids_asset_id    BIGINT REFERENCES data_asset(id),
    enabled                     BOOLEAN NOT NULL,
    localization_type           TEXT,
    localization_settings       JSON NOT NULL DEFAULT '{}',
    relationship_hash           TEXT NOT NULL,
    UNIQUE (plan_id, parameter_group_id, observation_group_id)
);
```

`source_rule_ids_asset_id` is optional provenance for plans where many rules
contributed to resolution. Small implementations may store that provenance in
`localization_settings` or omit it.

At large scale, UI workflows should query this resolved relationship set by
parameter group, observation group, parameter type, rule source, validation
status, or conflict status. Summary counts and representative samples are more
useful than rendering millions of disabled cells.

`localization_type` is meaningful only for `ES_MDA` plans. In an `ENIF_MDA`
plan it should be `NULL`, because EnIF-MDA is the plan-level algorithm and
applies consistently to all enabled parameter types.

Typical `ES_MDA` usage:

- `DISTANCE_LOCALIZATION` for fields and surfaces.
- `ADAPTIVE_LOCALIZATION` for scalars.
- `NO_LOCALIZATION` for global ES-MDA behavior where appropriate.

### Resolved Update Coverage

`resolved_update_pair` records the resolved coverage for one enabled resolved
relationship in one assimilation step. It references compact assets instead of
expanding every field or surface atom into relational rows.

```sql
CREATE TABLE resolved_update_pair (
    id                         BIGINT PRIMARY KEY,
    step_id                    BIGINT NOT NULL REFERENCES assimilation_step(id),
    relationship_id            BIGINT NOT NULL REFERENCES resolved_update_relationship(id),
    algorithm_type             TEXT NOT NULL,
    localization_type          TEXT,
    parameter_group_id         BIGINT NOT NULL REFERENCES parameter_group(id),
    observation_group_id       BIGINT NOT NULL REFERENCES observation_group(id),
    parameter_coverage_kind    TEXT NOT NULL CHECK (
                                   parameter_coverage_kind IN (
                                       'SCALAR_ROWS',
                                       'ATOM_MASK_ASSET'
                                   )
                               ),
    parameter_mask_asset_id    BIGINT REFERENCES data_asset(id),
    parameter_atom_count       BIGINT NOT NULL,
    parameter_membership_hash  TEXT NOT NULL,
    observation_set_asset_id   BIGINT REFERENCES data_asset(id),
    observation_count          BIGINT NOT NULL,
    observation_membership_hash TEXT NOT NULL,
    effective_settings         JSON NOT NULL DEFAULT '{}',
    UNIQUE (step_id, relationship_id)
);
```

For scalar groups, direct edge materialization gives a strong database-level
guardrail.

```sql
CREATE TABLE resolved_scalar_update_edge (
    step_id                BIGINT NOT NULL REFERENCES assimilation_step(id),
    relationship_id        BIGINT NOT NULL REFERENCES resolved_update_relationship(id),
    scalar_atom_id         BIGINT NOT NULL REFERENCES scalar_parameter_atom(id),
    observation_id         BIGINT NOT NULL REFERENCES observation(id),
    algorithm_type         TEXT NOT NULL,
    localization_type      TEXT,
    effective_settings     JSON NOT NULL DEFAULT '{}',
    PRIMARY KEY (step_id, scalar_atom_id, observation_id)
);
```

The primary key prevents the specific scalar conflict reported by the user: a
scalar parameter being updated by the same observation through multiple enabled
observation groups in the same assimilation step.

For fields and surfaces, the same invariant is enforced by validating overlaps
between mask assets and observation sets before execution.

### Validation Results

Validation produces durable records tied to the content hashes of the resolved
artifacts.

```sql
CREATE TABLE update_plan_validation (
    id                     BIGINT PRIMARY KEY,
    plan_id                BIGINT NOT NULL REFERENCES update_plan(id),
    step_id                BIGINT REFERENCES assimilation_step(id),
    status                 TEXT NOT NULL CHECK (
                               status IN ('VALID', 'INVALID', 'STALE')
                           ),
    validated_at           TIMESTAMP NOT NULL,
    validation_hash        TEXT NOT NULL,
    message                TEXT
);

CREATE TABLE update_relationship_conflict (
    id                         BIGINT PRIMARY KEY,
    validation_id              BIGINT NOT NULL REFERENCES update_plan_validation(id),
    left_relationship_id       BIGINT NOT NULL REFERENCES resolved_update_relationship(id),
    right_relationship_id      BIGINT NOT NULL REFERENCES resolved_update_relationship(id),
    parameter_overlap_count    BIGINT NOT NULL,
    observation_overlap_count  BIGINT NOT NULL,
    sample_conflicts_asset_id  BIGINT REFERENCES data_asset(id),
    message                    TEXT NOT NULL,
    CHECK (parameter_overlap_count > 0),
    CHECK (observation_overlap_count > 0)
);
```

A plan is executable only when the latest validation for every relevant step is
`VALID` and the validation hash still matches the resolved group and
relationship artifacts.

## Resolution Semantics

Resolution should be deterministic:

1. Resolve parameter groups from `parameter_group_selector` into scalar member
   rows or field/surface mask assets.
2. Resolve observation groups from `observation_group_rule` into observation
   member rows or observation-set assets.
3. Resolve `update_pair_rule` rows and explicit `update_pair` overrides into
   `resolved_update_relationship` rows. The default relationship state is
   disabled.
4. For every enabled resolved relationship, create one `resolved_update_pair` for
   each MDA step where the relationship participates.
5. For scalar relationships, populate `resolved_scalar_update_edge`.
6. Validate all enabled relationships for method compatibility, selector
   consistency, and overlap conflicts.
7. Mark the plan executable only after successful validation.

Selector rule semantics:

- Rules are evaluated within one group only.
- `rule_order` defines precedence.
- Later matching rules override earlier matching rules for the columns they set.
- `enabled_override = NULL` leaves the current enabled state unchanged.
- `enabled_override = TRUE` includes matching atoms or observations.
- `enabled_override = FALSE` excludes matching atoms or observations.
- Before the first matching include rule, membership defaults to excluded.
- `radius_override = NULL` leaves the current radius unchanged.
- If no matching observation rule provides a radius override, the effective
  radius remains `observation.default_radius`.
- A selector that never matches contributes nothing.
- Cross-group overlaps are not resolved by precedence. They are allowed only if
  they do not violate the update relationship overlap rules.

Relationship rule semantics:

- Relationship resolution starts from default deny.
- `update_pair_rule.rule_order` defines precedence between rules.
- Later matching relationship rules override earlier matching rules for the
  columns they set.
- `enabled_override = NULL` leaves the current relationship enabled state
  unchanged.
- `enabled_override = TRUE` enables matching relationships.
- `enabled_override = FALSE` disables matching relationships.
- Explicit `update_pair` rows override relationship rules for the matching
  parameter-group and observation-group pair.
- Disabled relationships do not produce `resolved_update_pair` coverage and do
  not participate in overlap validation.
- Resolution should not materialize the full Cartesian product of all parameter
  groups and observation groups merely to represent absence. It should enumerate
  matched relationships, explicit overrides, and compact diagnostics.

## MDA Overlap Rules

The hard invariant is checked per `assimilation_step`:

```text
For any two enabled resolved relationships A and B in the same step:

if parameter_atoms(A) intersects parameter_atoms(B)
and observations(A) intersects observations(B)
then the plan is invalid.
```

Consequences:

- The same observation may update `PORO` and `PERM` if those are distinct
  parameter atoms.
- The same observation may update a scalar and a field cell if those are
  distinct parameter atoms.
- Two observation groups may overlap globally.
- Two parameter groups may overlap globally.
- Overlap becomes invalid only when both the parameter coverage and observation
  coverage overlap for enabled relationships in the same step.
- Disabled relationships do not participate in overlap validation.
- No cross-relationship precedence rule may silently choose one relationship over
  another. Cross-relationship conflicts must be reported as invalid
  configuration.

Validation should scale with the number of enabled resolved relationships and
the size of their resolved artifacts, not with the full number of possible group
combinations. Implementations should use hashes, sorted membership assets,
bitsets, bloom filters, spatial indexes, interval indexes, or chunked mask
intersection to avoid naive all-row expansion. Conflict reports should store
counts and representative samples, not every conflicting atom-observation edge.

Additional execution-safety rules:

- `ENIF_MDA` is selected at the plan level and applies to all enabled parameter
  groups and observation groups.
- `ES_MDA` is selected at the plan level. Relationship-level settings may choose
  localization behavior, but they do not switch the plan into EnIF-MDA.
- If the same parameter atom is enabled for multiple disjoint observation groups
  in the same ES-MDA assimilation step, those observations should be assembled
  into one update for that atom under the plan-level `ES_MDA` algorithm.
- If the same parameter atom has incompatible ES-MDA localization settings
  across disjoint observation groups, validation should reject the plan unless a
  mathematically well-defined combined interpretation exists.

## Algorithm And Localization Dependencies

`algorithm_type` determines the MDA algorithm for the whole plan.
`localization_type` is only applicable when `algorithm_type = 'ES_MDA'`.

| Plan algorithm | Scalars | Fields | Surfaces | Required metadata |
| --- | --- | --- | --- | --- |
| `ES_MDA` | Yes | Yes | Yes | Ensemble responses, observation errors, ES-MDA step scaling |
| `ENIF_MDA` | Yes | Yes | Yes | Parameter graph or precision structure, response matrix, EnIF-MDA settings |

| ES-MDA localization | Scalars | Fields | Surfaces | Required metadata |
| --- | --- | --- | --- | --- |
| `ADAPTIVE_LOCALIZATION` | Yes | Possible | Possible | Ensemble correlations, observation errors, correlation threshold settings |
| `DISTANCE_LOCALIZATION` | Usually no | Yes | Yes | Parameter coordinates, observation coordinates, radius settings |
| `NO_LOCALIZATION` | Yes | Yes | Yes | Ensemble responses and observation errors |

Logical dependencies:

- `enabled = FALSE` means the relationship contributes no resolved update
  coverage.
- `update_plan.algorithm_type` must be either `ES_MDA` or `ENIF_MDA`.
- In an `ES_MDA` plan, each enabled relationship must have a valid
  `localization_type`.
- In an `ENIF_MDA` plan, enabled relationships should have
  `localization_type = NULL`.
- Localization-specific settings belong in
  `update_pair.localization_settings`, `update_pair_rule.localization_settings`,
  `resolved_update_relationship.localization_settings`, or
  `resolved_update_pair.effective_settings`.
- Observation-specific settings such as effective radius belong on observation
  group rules or resolved observation membership, not on `observation` itself.
- EnIF-MDA requires a graph or precision model. For scalars, that may be an
  independent-node graph, a user-provided dependency graph, or a learned graph.
- Distance localization requires parameter and observation coordinates. It
  should fail validation if those coordinates are missing for required atoms or
  observations.
- Any change to selectors, observations, parameter geometry, active masks, or
  algorithm/localization settings invalidates prior resolution and validation
  hashes.

## Bulk Artifact Conventions

Resolved mask and set assets should be self-describing and content-addressable
through `data_asset.content_hash`.

Recommended sparse parameter mask columns:

```text
parameter_id, atom_index
```

Optional diagnostic columns:

```text
i, j, k, x, y, z, active
```

Recommended observation set columns:

```text
observation_id
```

Optional diagnostic columns:

```text
observation_key, obs_index, response_key, east, north, time_value
```

Conflict sample assets should contain enough information for a UI to explain
why a plan is invalid without loading full field or surface masks.

## Example

Example groups:

| Group | Type | Meaning |
| --- | --- | --- |
| `poro_scalars` | `SCALAR` | Selected scalar porosity multipliers |
| `poro_field_region_a` | `FIELD` | Active `PORO` cells inside region A |
| `top_surface_fault_block` | `SURFACE` | Surface nodes inside a fault block |
| `well_pressures` | observation group | Pressure observations |
| `well_rates` | observation group | Rate observations |

Example sparse `ES_MDA` update relationships:

| Parameter group | Observation group | Enabled | Localization |
| --- | --- | --- | --- |
| `poro_scalars` | `well_pressures` | `TRUE` | `ADAPTIVE_LOCALIZATION` |
| `poro_scalars` | `well_rates` | `TRUE` | `ADAPTIVE_LOCALIZATION` |
| `poro_field_region_a` | `well_pressures` | `TRUE` | `DISTANCE_LOCALIZATION` |
| `top_surface_fault_block` | `well_rates` | `FALSE` | `NULL` |

The first three rows may be explicit `update_pair` rows or the result of
`update_pair_rule` resolution. The disabled row is only useful if it records an
explicit deny, for example an exception to a rule that would otherwise enable
all surface groups for rate observations.

This relationship set is valid only if `well_pressures` and `well_rates` resolve
to disjoint observation datums for `poro_scalars`, or if the overlapping scalar
atoms are otherwise not shared between the enabled relationships. If an
observation is present in both `well_pressures` and `well_rates`, and both
relationships update the same scalar atom in the same assimilation step,
validation must reject the plan.

The same pressure observation may still update `poro_scalars` and
`poro_field_region_a` if the parameter atoms are distinct.

For an `ENIF_MDA` plan, the same sparse relationship model would still control
which relationships are enabled, but the localization column would be `NULL` for
all enabled relationships. EnIF-MDA is then applied as the plan-level algorithm to
all enabled parameter types.

## Design Constraints

- Do not store field or surface values as one relational row per cell.
- Do not require one relational row per field or surface atom in a resolved
  update edge table.
- Do not require one relational row for every possible parameter-group and
  observation-group combination. Store enabled relationships, explicit denies,
  and rule-derived resolved relationships sparsely.
- Store large values, masks, response matrices, and resolved coverage sets as
  external assets with explicit format, schema version, count, and hash.
- Keep the relational database authoritative for configuration, provenance,
  validation status, and asset references.
- Keep bulk assets authoritative for large arrays and large resolved masks.
- Relationship-specific localization settings belong on `update_pair`,
  `update_pair_rule`, or resolved relationship artifacts, not on the base
  parameter or observation facts.
- Group selector precedence is local to one group. Cross-relationship conflicts
  are validation errors.
- The executable representation is the resolved and validated update coverage,
  not the raw selector rules.
- The plan must fail closed: if a required artifact is missing, stale, or has a
  mismatching hash, the update must not run.

## Project, Experiment, and Pipeline Model

The update plan describes *how* parameters are updated by observations. The
project, experiment, and pipeline model describes *what* is being run and *in
what order*. It is the orchestration layer that connects parameter
configuration, observation configuration, update plans, and forward model
execution into a coherent workflow.

The update plan is algorithm configuration (ES-MDA with assimilation steps and
localization rules). The pipeline is workflow orchestration (which steps run, in
what order, producing which ensembles).

### Modeling Goals

- Support all experiment types: ensemble experiments, history matching (ES-MDA
  with any number of weights), sensitivity analysis, optimization (Everest),
  prediction/forecast runs, manual updates, and re-evaluation.
- Treat sensitivity analysis as a simulator-response workflow, not an
  observation-assimilation workflow. It studies synthetic responses produced by
  `EVALUATE` or `PREDICT` steps and does not require real observations or an
  update plan.
- Replace hardcoded `RunModel` class hierarchies with a configurable pipeline
  stored in the relational database.
- Allow users to compose pipeline steps without modifying code.
- Keep the pipeline model generic enough that ES-MDA with one weight is the same
  workflow as ES-MDA with multiple weights — only the number of assimilation
  steps differs.
- Provide project-level organization so that experiments can reference each
  other (e.g., a prediction experiment consuming a history-matching posterior).
- Track execution status, provenance, and timing at the step and ensemble level.
- Keep the pipeline schema small: the pipeline is a sequence of steps that
  reference existing configuration objects (update plans, parameter configs,
  observation configs, forward model configs), not a complete workflow engine.

### Core Concepts

#### Project

A named organizational container for related experiments. A project may contain
history-matching studies, optimization runs, sensitivity analyses, and
prediction forecasts that share a common geological model or field. Projects
enable cross-experiment references and project-level metadata and search.

#### Experiment

One coherent study. An experiment has a type that describes its high-level
purpose, but the actual workflow is defined by its pipeline. The experiment
references the update plan it uses (if any), the parameter/observation
configuration, and the forward model settings. An experiment's status tracks its
lifecycle: draft, pending, running, completed, failed, or cancelled.

Experiments within a project may reference each other's ensembles. For example,
a prediction experiment may declare its input as one of the posterior ensembles
from a completed history-matching experiment.

#### Pipeline

An ordered sequence of steps that defines the experiment's workflow. Most
pipelines are simple chains where each step consumes the previous step's output
and produces input for the next step. The pipeline is the configurable
replacement for hardcoded `RunModel` subclasses.

#### Pipeline Step

One atomic operation in the pipeline. Each step has a type, a predecessor link,
and step-specific settings. Steps are executed sequentially. At execution time,
each step populates `source_ensemble_id` (the ensemble it reads) and
`target_ensemble_id` (the ensemble it writes to).

Step types:

| Step Type | Creates ensemble? | Input | Output | Description |
| --- | --- | --- | --- | --- |
| `SAMPLE` | Yes | — | Ensemble with sampled parameters | Sample parameters from prior distributions into a new ensemble |
| `EVALUATE` | No | Ensemble with parameters | Same ensemble, now with responses | Run forward model, store responses in the same ensemble |
| `UPDATE` | Yes | Ensemble with parameters and responses | New ensemble with updated parameters | Run assimilation algorithm, producing a posterior ensemble |
| `PREDICT` | No | Ensemble with parameters | Same ensemble, now with forecast responses | Evaluate forward model with prediction-horizon settings |
| `OPTIMIZE` | Yes (multiple) | — | Last batch ensemble | Run Everest-style optimization loop |
| `ANALYZE` | No | Ensemble with responses | Analysis results (asset) | Run sensitivity analysis or post-update diagnostics |
| `MANUAL_EDIT` | Yes | Ensemble | New ensemble with modified parameters | Create an ensemble copy with user-driven parameter overrides |
| `LOAD` | No | Existing ensemble | Same ensemble (referenced) | Import an ensemble from another experiment as pipeline input |

#### Ensemble

One set of realizations produced by one evaluation (or sampling) step. An
ensemble belongs to exactly one experiment and is created by exactly one
pipeline step. An ensemble may declare a `prior_ensemble_id` when it was
produced by an update step, linking the posterior to its prior. This link forms
the update chain and is needed for provenance and iteration tracking.

Large parameter and response values are not stored in the ensemble row. They
live in `parameter_value_asset` and `response_value_asset` (defined in the
update plan section), keyed by `ensemble_id`.

### Relational Tables

The SQL below extends the schema from the update plan section. `data_asset` and
all previously defined tables are shared.

```sql
-- project: top-level organizational container
CREATE TABLE project (
    id            BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata      JSONB NOT NULL DEFAULT '{}'
);

-- experiment: one coherent study
CREATE TABLE experiment (
    id              BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    project_id      BIGINT NOT NULL REFERENCES project(id),
    name            TEXT NOT NULL,
    description     TEXT,
    experiment_type TEXT NOT NULL CHECK (experiment_type IN (
                        'ENSEMBLE_EXPERIMENT',
                        'HISTORY_MATCHING',
                        'SENSITIVITY_ANALYSIS',
                        'OPTIMIZATION',
                        'PREDICTION',
                        'MANUAL_UPDATE',
                        'EVALUATE'
                    )),
    update_plan_id  BIGINT REFERENCES update_plan(id),
    status          TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN (
                        'draft', 'pending', 'running',
                        'completed', 'failed', 'cancelled'
                    )),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata        JSONB NOT NULL DEFAULT '{}',
    UNIQUE (project_id, name)
);

-- pipeline: ordered steps defining experiment workflow
CREATE TABLE pipeline (
    id              BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    experiment_id   BIGINT NOT NULL REFERENCES experiment(id),
    name            TEXT NOT NULL DEFAULT 'default',
    description     TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, name)
);

-- pipeline_step: one atomic operation in the workflow
CREATE TABLE pipeline_step (
    id                      BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    pipeline_id             BIGINT NOT NULL REFERENCES pipeline(id)
                            ON DELETE CASCADE,
    step_index              INTEGER NOT NULL,
    step_type               TEXT NOT NULL CHECK (step_type IN (
                                'SAMPLE',
                                'EVALUATE',
                                'UPDATE',
                                'PREDICT',
                                'OPTIMIZE',
                                'ANALYZE',
                                'MANUAL_EDIT',
                                'LOAD'
                            )),
    label                   TEXT NOT NULL,
    predecessor_step_id     BIGINT REFERENCES pipeline_step(id),

    -- Populated at execution time:
    source_ensemble_id      BIGINT REFERENCES ensemble(id),
    target_ensemble_id      BIGINT REFERENCES ensemble(id),

    -- For UPDATE steps:
    update_plan_id          BIGINT REFERENCES update_plan(id),
    assimilation_step_index INTEGER,

    status                  TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN (
                                'pending', 'running', 'completed',
                                'failed', 'skipped', 'blocked'
                            )),
    settings                JSONB NOT NULL DEFAULT '{}',
    UNIQUE (pipeline_id, step_index)
);

-- ensemble: one set of realizations from one pipeline step
CREATE TABLE ensemble (
    id                BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    experiment_id     BIGINT NOT NULL REFERENCES experiment(id),
    pipeline_step_id  BIGINT NOT NULL REFERENCES pipeline_step(id),
    prior_ensemble_id BIGINT REFERENCES ensemble(id),
    name              TEXT NOT NULL,
    iteration         INTEGER NOT NULL DEFAULT 0,
    ensemble_size     INTEGER NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN (
                          'pending', 'running',
                          'completed', 'failed'
                      )),
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ,
    metadata          JSONB NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, name)
);

-- analysis_result: metadata for outputs from ANALYZE steps
CREATE TABLE analysis_result (
    id                    BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    experiment_id         BIGINT NOT NULL REFERENCES experiment(id),
    pipeline_step_id      BIGINT NOT NULL REFERENCES pipeline_step(id),
    source_ensemble_id    BIGINT NOT NULL REFERENCES ensemble(id),
    result_type           TEXT NOT NULL CHECK (result_type IN (
                              'SENSITIVITY',
                              'CORRELATION',
                              'DESIGN_MATRIX_DIAGNOSTIC',
                              'POST_UPDATE_DIAGNOSTIC'
                          )),
    result_asset_id       BIGINT NOT NULL REFERENCES data_asset(id),
    design_asset_id       BIGINT REFERENCES data_asset(id),
    parameter_axis_hash   TEXT,
    response_axis_hash    TEXT,
    result_hash           TEXT NOT NULL,
    settings              JSONB NOT NULL DEFAULT '{}',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (pipeline_step_id, result_type, result_hash)
);
```

Notes:

- `project`, `experiment`, and `pipeline` use `GENERATED ALWAYS AS IDENTITY` for
  surrogate keys. The update plan tables previously used manually assigned
  `BIGINT PRIMARY KEY`. For consistency, all new tables use identity columns.
  Update plan tables may migrate to identity columns as well.
- `pipeline_step.predecessor_step_id` forms the step chain. For a simple linear
  pipeline, each step's predecessor is the immediately preceding step. This
  column also supports future branching (joining two pipelines or inserting
  conditional steps).
- `pipeline_step.update_plan_id` and `assimilation_step_index` are non-null only
  for `UPDATE` steps. They select which update plan and which specific
  assimilation step to execute.
- `experiment.update_plan_id` is nullable because not every experiment performs
  assimilation. A `SENSITIVITY_ANALYSIS` experiment normally leaves it `NULL`;
  its pipeline samples or perturbs parameters, evaluates the simulator, and
  analyzes synthetic responses without matching against observations.
- `pipeline_step.source_ensemble_id` and `target_ensemble_id` are populated
  during execution. At configuration time they are `NULL`; the pipeline is
  defined structurally (step type and predecessor) and the IDs are resolved at
  runtime.
- `ensemble.prior_ensemble_id` links an update step's output (posterior) to its
  input (prior). This is the `prior_ensemble_id` that already exists in the
  current ERT storage model.
- `ensemble.iteration` tracks the iteration number within a multi-step workflow.
  For ES-MDA, iteration 0 is the prior ensemble, iteration 1 is the first
  posterior, etc. For optimization, iteration maps to batch number.
- `analysis_result` records queryable metadata for outputs from `ANALYZE` steps.
  The bulk result remains an external asset so large sensitivity matrices,
  Sobol indices, screening metrics, correlation tables, or diagnostic plots do
  not become relational rows.

### Pipeline Step Semantics

Each step type has well-defined behavior for input resolution, execution, and
output.

#### SAMPLE

Creates a new ensemble and populates it with parameters sampled from prior
distributions.

Execution:

1. Create a new `ensemble` row with `pipeline_step_id = current_step`,
   `prior_ensemble_id = NULL`, `iteration` determined by the step's position in
   the pipeline.
2. For each parameter in the experiment's parameter configuration, sample values
   from its prior distribution for each active realization.
3. Store sampled values in `parameter_value_asset` rows keyed by the new
   ensemble.
4. Set `target_ensemble_id` to the new ensemble.

Settings:

- `realizations`: optional mask of active realizations (default: all)
- `random_seed`: optional seed override (default: experiment seed)
- `design_matrix`: optional design matrix for parameter variations

#### EVALUATE

Runs the forward model for every active realization in the source ensemble and
stores the resulting responses.

Execution:

1. Read the source ensemble (from `predecessor_step_id`'s target) to obtain
   parameter values.
2. Create runpaths and forward model input files from parameter values.
3. Run the forward model through the ensemble evaluator.
4. Read response values from forward model output files.
5. Store responses in `response_value_asset` rows keyed by the source ensemble.
6. Set `source_ensemble_id = predecessor.target_ensemble_id`,
   `target_ensemble_id = source_ensemble_id` (responses are stored in-place in
   the source ensemble).

Settings:

- `realizations`: optional active realization mask
- `forward_model_steps`: optional override of forward model steps
- `runpath_format`: optional runpath format string override

#### UPDATE

Runs one assimilation step from the referenced update plan, creating a new
posterior ensemble with updated parameter values.

Execution:

1. Read the source ensemble (prior) to obtain prior parameter values and
   responses.
2. Verify that the source ensemble has the required responses for the
   observations referenced in the update plan.
3. Run the assimilation algorithm (ES-MDA or EnIF-MDA) specified by the update
   plan for the given `assimilation_step_index`.
4. Create a new `ensemble` row with `prior_ensemble_id = source_ensemble.id`.
5. Store the updated parameter values in `parameter_value_asset` rows keyed by
   the new ensemble.
6. Set `source_ensemble_id = source_ensemble.id`,
   `target_ensemble_id = new_ensemble.id`.

The `assimilation_step_index` selects which weight or scaling to apply from the
update plan's assimilation steps. For ES-MDA with one weight, the update plan
has a single assimilation step and the pipeline contains one `UPDATE` step with
`assimilation_step_index = 0`. For multi-step ES-MDA, the pipeline contains one
`UPDATE` step per assimilation step, each with a different index. ES is ES-MDA
with a single weight — there is no separate ES algorithm.

Settings:

- `weight`: optional override of the assimilation step's scaling factor

#### PREDICT

Like `EVALUATE`, but with a prediction or forecast forward model. The source
ensemble typically comes from the last `UPDATE` step's posterior.

Execution: identical to `EVALUATE`, but the settings may specify a longer
simulation horizon, additional output keys, or a different forward model
configuration.

Settings:

- `prediction_horizon`: time, date, or step count for the forecast
- `forward_model_steps`: prediction-specific forward model steps (may differ
  from the base evaluation)
- `response_keys`: additional response keys to collect during forecast

#### OPTIMIZE

Runs the Everest optimization loop as a single pipeline step. Internally, the
step expands into a sequence of batches:

1. Initialize optimizer (ropt) with the experiment's optimization configuration
   (controls, objective functions, constraints, algorithm settings).
2. For each batch:
   a. Generate control vectors for this batch (from optimizer).
   b. Create a batch ensemble (iteration = batch number) with control values as
      parameters.
   c. Evaluate the batch ensemble (run forward model).
   d. Read objective and constraint values from forward model output.
   e. Feed results to the optimizer.
   f. Check termination criteria (max batches, convergence).
3. The final batch ensemble is the step's `target_ensemble_id`.

The optimization configuration is stored in `settings` as a structured JSONB
document:

- `optimizer_algorithm`: e.g., `optpp_q_newton`, `scipy_slsqp`
- `max_batches`: termination condition
- `max_function_evaluations`: termination condition
- `tolerance`: convergence tolerance
- `controls`: list of control group configurations (each with name, variables,
  bounds, initial guesses, perturbation magnitudes)
- `objective_functions`: list of objective function configurations (each with
  name, weight, scale, aggregation type)
- `input_constraints`: linear constraints on controls
- `output_constraints`: nonlinear constraint configurations
- `cvar`: optional CVaR risk settings
- `auto_scale`: whether to auto-scale objectives and constraints from first
  batch data

Because the optimization loop may create tens to hundreds of batch ensembles,
the pipeline step does not create a separate `pipeline_step` row per batch.
Instead, batch ensembles are linked to the `OPTIMIZE` step through
`ensemble.pipeline_step_id` and are differentiated by `ensemble.iteration`.

#### ANALYZE

Runs post-evaluation diagnostics such as sensitivity analysis,
parameter-response correlation analysis, or post-update diagnostics. The step
reads parameter values and synthetic response values from the source ensemble and
writes one or more `analysis_result` rows whose bulk outputs are `data_asset`
objects.

For sensitivity analysis, `ANALYZE` does not read real observations,
observation groups, or update-plan coverage. It analyzes the simulator responses
stored in `response_value_asset` for the source ensemble against the sampled or
perturbed parameter values stored in `parameter_value_asset`. If the analysis
depends on an explicit design matrix, that design is referenced by
`analysis_result.design_asset_id` or by the `SAMPLE` step settings.

Settings:

- `analysis_type`: e.g., `SENSITIVITY`, `DESIGN_MATRIX_DIAGNOSTIC`,
  `CORRELATION`, `POST_UPDATE_DIAGNOSTIC`
- `method`: e.g., `SOBOL`, `MORRIS`, `OAT`, `REGRESSION`, `CORRELATION`
- `parameter_selectors`: which parameters to analyze
- `response_selectors`: which responses to analyze
- `design_asset_id`: optional explicit design matrix or sensitivity design
- `result_metrics`: requested metrics, such as first-order indices, total-order
  indices, standardized regression coefficients, rank correlations, or tornado
  data

#### MANUAL_EDIT

Creates a copy of the source ensemble with user-specified parameter
modifications. This enables manual update workflows where the user reviews and
adjusts parameters before re-evaluation.

Execution:

1. Create a new `ensemble` row (copy of source).
2. The user provides a set of parameter-value overrides.
3. Store the modified parameter values in `parameter_value_asset` rows keyed by
   the new ensemble.
4. Set `source_ensemble_id = source_ensemble.id`,
   `target_ensemble_id = new_ensemble.id`.

Settings:

- `overrides`: user-provided parameter modifications (stored as JSON or as an
  asset reference for large overrides)

#### LOAD

Imports an ensemble from another experiment into the current pipeline. This
enables cross-experiment workflows such as running a prediction on a posterior
from a completed history-matching experiment.

Execution:

1. Resolve the source experiment and ensemble by ID or name.
2. Reference or copy the existing ensemble row and its parameter and response
   assets.
3. Set `source_ensemble_id = existing_ensemble.id`,
   `target_ensemble_id = source_ensemble_id`.

Settings:

- `source_experiment_id`: the experiment containing the source ensemble
- `source_ensemble_identifier`: ensemble ID or name within the source experiment

#### Type Compatibility

Not all step type sequences are valid. The table below shows valid predecessor
to step transitions:

| Predecessor | Step | Valid? | Notes |
| --- | --- | --- | --- |
| (none) | `SAMPLE` | Yes | Start of a history-matching or ensemble-experiment pipeline |
| (none) | `LOAD` | Yes | Start of a prediction pipeline consuming an external posterior |
| (none) | `OPTIMIZE` | Yes | Start of an optimization pipeline |
| (none) | `EVALUATE` | No | No parameters to evaluate |
| (none) | `UPDATE` | No | No prior ensemble to update |
| `SAMPLE` | `EVALUATE` | Yes | Evaluate prior parameters |
| `LOAD` | `EVALUATE` | Yes | Evaluate loaded parameters |
| `LOAD` | `PREDICT` | Yes | Forecast on loaded ensemble |
| `EVALUATE` | `UPDATE` | Yes | Update from evaluated prior |
| `UPDATE` | `EVALUATE` | Yes | Evaluate posterior (standard after update) |
| `EVALUATE` | `PREDICT` | Yes | Run forecast on evaluated ensemble |
| `UPDATE` | `PREDICT` | Yes | Run forecast on posterior without re-evaluation |
| `UPDATE` | `UPDATE` | Yes | Chained MDA updates (ES-MDA: one `UPDATE` per weight) |
| `EVALUATE` | `ANALYZE` | Yes | Analyze responses |
| `UPDATE` | `ANALYZE` | Yes | Analyze posterior responses |
| `EVALUATE` | `MANUAL_EDIT` | Yes | Edit parameters based on results |
| `MANUAL_EDIT` | `EVALUATE` | Yes | Evaluate manually modified parameters |
| `OPTIMIZE` | `EVALUATE` | Yes | Evaluate optimal controls (post-optimization) |
| `OPTIMIZE` | `PREDICT` | Yes | Forecast from optimal controls |

Implementations should validate the full pipeline chain at configuration time
and report invalid sequences.

### Experiment Types and Pipeline Templates

An experiment type is a high-level label. On creation, the system generates a
default pipeline template for the type. The user may then customize the pipeline
(add, remove, reorder, or reconfigure steps).

The same experiment type may produce different pipeline lengths depending on
configuration. For example, `HISTORY_MATCHING` with an ES-MDA update plan
produces one `UPDATE` and `EVALUATE` pair per assimilation step. A single-weight
ES-MDA plan produces one pair; a multi-weight plan produces multiple pairs. ES
is ES-MDA with one weight.

#### ENSEMBLE_EXPERIMENT

Purpose: Sample parameters from prior distributions and evaluate all
realizations once.

Default pipeline:

```
 Step 0: SAMPLE   → ensemble "prior"
 Step 1: EVALUATE → ensemble "prior" (responses stored in-place)
```

#### HISTORY_MATCHING

Purpose: Calibrate parameters to observations using ES-MDA (one or more
weights). Covers both Ensemble Smoother (ES, one weight of 1.0) and ES-MDA
(multiple weights).

Default pipeline for single-weight ES-MDA (ES):

```
 Step 0: SAMPLE             → ensemble "prior"
 Step 1: EVALUATE           → ensemble "prior"
 Step 2: UPDATE (index 0)   → ensemble "posterior_0"
 Step 3: EVALUATE           → ensemble "posterior_0"
```

Default pipeline for N-step ES-MDA:

```
 Step 0: SAMPLE                  → ensemble "prior"
 Step 1: EVALUATE                → ensemble "prior"
 Step 2: UPDATE (index 0)        → ensemble "posterior_0"
 Step 3: EVALUATE                → ensemble "posterior_0"
 Step 4: UPDATE (index 1)        → ensemble "posterior_1"
 Step 5: EVALUATE                → ensemble "posterior_1"
 ...
 Step N*2:   UPDATE (index M)    → ensemble "posterior_M"
 Step N*2+1: EVALUATE            → ensemble "posterior_M"
```

The update plan's `assimilation_step` rows define the weights and their count.
The pipeline template reads the count from the referenced update plan and
generates `UPDATE` and `EVALUATE` pairs accordingly. ES is history matching with
an ES-MDA update plan containing a single assimilation step. No separate
`ENSEMBLE_SMOOTHER` experiment type is needed.

#### SENSITIVITY_ANALYSIS

Purpose: Study how parameter variations affect responses using design matrices,
one-at-a-time perturbations, or other sensitivity methods.

Sensitivity analysis is independent of real-life observations. It does not need
`observation`, `observation_group`, `update_plan`, `assimilation_step`, or
resolved update coverage unless a custom workflow explicitly adds an update
step. The normal workflow is to generate a parameter design, evaluate the
simulator for that design, and compute sensitivity metrics from the synthetic
responses.

Default pipeline:

```
 Step 0: SAMPLE (with design matrix) → ensemble "design"
 Step 1: EVALUATE                    → ensemble "design"
 Step 2: ANALYZE                     → analysis results
```

The design matrix or other sensitivity configuration is specified in the
`SAMPLE` step's settings, the experiment's parameter configuration, or an
external `DESIGN_MATRIX` or `SENSITIVITY_DESIGN` asset. The `ANALYZE` step
records its outputs in `analysis_result` with `result_type = 'SENSITIVITY'` and
stores bulk results in a `SENSITIVITY_RESULT` or `ANALYSIS_RESULT` asset.

Typical sensitivity outputs include Sobol indices, Morris screening scores,
one-at-a-time response deltas, parameter-response correlations, regression
coefficients, and response summary statistics grouped by parameter level.

#### OPTIMIZATION

Purpose: Find optimal control values using the Everest and ropt optimization
loop. Maximizes objective functions subject to input and output constraints.

Default pipeline:

```
 Step 0: OPTIMIZE → final batch ensemble
```

Post-optimization evaluation may be added as additional steps:

```
 Step 0: OPTIMIZE       → final batch ensemble
 Step 1: EVALUATE       → final batch ensemble (re-evaluate optimal controls)
 Step 2: PREDICT        → final batch ensemble (optional forecast)
```

#### PREDICTION

Purpose: Run forecasts from an existing posterior ensemble.

Default pipeline:

```
 Step 0: LOAD (external posterior) → loaded ensemble
 Step 1: PREDICT                   → same ensemble (forecast responses added)
```

The `LOAD` step references a completed history-matching experiment and its
posterior ensemble by experiment ID and ensemble name.

#### MANUAL_UPDATE

Purpose: Copy an existing ensemble, apply user-specified parameter
modifications, and optionally re-evaluate.

Default pipeline:

```
 Step 0: LOAD (or any ensemble-producing step)
 Step 1: MANUAL_EDIT → modified ensemble
 Step 2: EVALUATE    → same ensemble (responses for modified parameters)
```

#### EVALUATE

Purpose: Re-evaluate an existing ensemble. Runs the forward model for an
ensemble that already has parameters but needs fresh responses.

Default pipeline:

```
 Step 0: LOAD → loaded ensemble
 Step 1: EVALUATE → same ensemble
```

### Pipeline Resolution and Validation

Pipeline configuration goes through two phases: design-time validation and
runtime resolution.

#### Design-Time Validation

1. **Structural validation**: All step references (`predecessor_step_id`) form a
   connected chain. No cycles.
2. **Type compatibility**: The sequence of step types follows the valid
   predecessor-to-step transition matrix.
3. **Update plan reference**: Every `UPDATE` step references an existing, valid
   update plan. The `assimilation_step_index` must be within the plan's
   assimilation step count.
4. **Ensemble naming**: No duplicate ensemble names within an experiment.
5. **Cross-experiment reference**: `LOAD` steps reference existing experiments
   and ensembles.

#### Runtime Resolution

1. **Source ensemble resolution**: For each step, `source_ensemble_id` is
   resolved from the predecessor step's `target_ensemble_id`. For steps with no
   predecessor (`SAMPLE`, `LOAD`, `OPTIMIZE`), the source is `NULL`.
2. **Target ensemble creation**:
   - Steps that create new ensembles (`SAMPLE`, `UPDATE`, `MANUAL_EDIT`,
     `OPTIMIZE`) generate a new `ensemble` row at execution start. The ensemble
     name is either user-specified in the step's settings or auto-generated from
     the step label and pipeline position.
   - Steps that evaluate in-place (`EVALUATE`, `PREDICT`) set
     `target_ensemble_id = source_ensemble_id`.
3. **Parameter and response reading**: Steps that evaluate ensembles read
   parameter values from `parameter_value_asset` rows keyed by the source
   ensemble. Steps that update parameters write new values to
   `parameter_value_asset` rows keyed by the target ensemble.
4. **UPDATE step algorithm resolution**: The `UPDATE` step reads the update
   plan's `resolved_update_pair` and `resolved_scalar_update_edge` tables (from
   the update plan section) to determine which parameters are updated by which
   observations and with which localization settings.

#### Execution State Machine

Each pipeline step follows this state machine:

```text
pending → running → completed
  ↓          ↓
blocked    failed
  ↓
skipped
```

- `pending`: not yet started
- `running`: executing (forward model running, update algorithm computing, etc.)
- `completed`: finished successfully; `target_ensemble_id` is populated
- `failed`: execution error; the pipeline is halted
- `blocked`: predecessor step failed; this step cannot execute
- `skipped`: manually skipped by the user

The pipeline as a whole is:
- `pending` when the experiment is created
- `running` when any step is running
- `completed` when all steps are completed or skipped
- `failed` when any step is failed

#### Provenance

The pipeline structure provides full provenance:

- `ensemble.pipeline_step_id` — which step created this ensemble
- `pipeline_step.predecessor_step_id` — which step preceded this one
- `ensemble.prior_ensemble_id` — which ensemble was the prior (for `UPDATE`
  steps)
- `pipeline_step.update_plan_id` and `assimilation_step_index` — which update
  algorithm was used

This chain enables the system to answer queries like:

- What update plan produced this posterior?
- Which prior ensemble was updated to create this ensemble?
- What pipeline configuration generated this result?

### Storage Split

The relational database stores:

- Project, experiment, pipeline, and pipeline step metadata.
- Ensemble metadata (name, size, status, timing).
- Links to parameter and response assets.
- Analysis result metadata and links to design/result assets.
- Execution status and provenance.

Object storage (existing, shared with update plan):

- Parameter ensemble values.
- Response matrices.
- Forecast and response values for prediction steps.
- Optimization batch results (objectives, constraints, gradients).
- Design matrices, sensitivity designs, and analysis results from `ANALYZE`
  steps.

### Design Constraints

- Do not store parameter or response values as relational rows. Ensemble value
  storage belongs to the update plan's `parameter_value_asset` and
  `response_value_asset` tables.
- Do not duplicate update plan configuration in pipeline steps. Pipeline steps
  reference update plans by ID.
- Do not require observations or update plans for sensitivity analysis.
  Sensitivity workflows consume parameter values and synthetic simulator
  responses from ensembles, plus optional design assets.
- Keep the pipeline model sparse. Most fields are optional and populate at
  execution time.
- Do not embed optimization orchestration logic (ropt, batch loop) in the
  pipeline schema. The `OPTIMIZE` step is a single step type whose internal loop
  is implementation-defined.
- A pipeline must have at least one step.
- An experiment must have exactly one active pipeline.
- A pipeline step must have exactly one predecessor, except for the first step.
- An ensemble must belong to exactly one experiment and one pipeline step.
- Prior ensemble references (`prior_ensemble_id`) must form a directed acyclic
  graph. No cycles in the update chain.
- Pipeline validation must happen before execution, not during. If a pipeline is
  invalid, the experiment status remains `draft`.
- Cross-experiment ensemble references (`LOAD` steps) must use content hashes or
  ensemble IDs to ensure reproducibility. If the source experiment or ensemble
  is deleted, the `LOAD` step becomes invalid.
