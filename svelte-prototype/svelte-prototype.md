# Update Strategy Setup - Code Investigation Findings

## Overview

This document summarizes the findings from investigating the ERT (Ensemble Reservoir Tool) codebase to inform the design of a new **Update Strategy Setup** user interface. The goal is to create a UI that allows users to:
1. Choose which algorithms to use for updating different parameters
2. Define distance-based localization properties per observation

## Design Inspiration

ResX (a competitor tool) has a tab called **"Define localization setup"** that provides capabilities we should consider for ERT's Update Strategy Setup:

![alt text](image-2.png)

### Observations from ResX Screenshot

**1. Per-Parameter Type Localization Tabs:**
- **Grid properties tab** (3D fields): Localization specified as distance-based with radius in metres (e.g., `r=1500 m`)
  - Maps directly to ERT's `FIELD` parameter type
- **Surfaces tab** (2D): Similar distance-based localization for surface parameters
  - Maps directly to ERT's `SURFACE` parameter type
- **Scalars tab** (1D): Some form of localization available for scalar parameters
  - Most likely adaptive localization (ResX uses adaptive for scalars)
  - ERT can offer both **adaptive localization** AND **EnIF** for scalar parameters (GenKw, EverestControl)

**2. Per-Observation Localization Configuration (Phase 4 in ERT roadmap):**
- Each **row is one observation** (well + data type combination)
- **Radius set independently per cell** - complete per-observation control
- ERT currently has NO per-observation equivalent:
  - `UPDATE:TRUE`/`UPDATE:FALSE` is a **per-parameter** flag on `FIELD`, `SURFACE`, `GEN_KW`
  - No `UPDATE:NONE` equivalent in ERT config
  - No way to disable individual observations from config file

**3. Independent Localization per Parameter:**
- Both petrophysical properties (PermeabilityXY, Porosity) and facies representations (Gaussian1, Gaussian2 — APS Gaussian random fields) can be updated **simultaneously**
- **Independently configured localization radii per parameter**
- This is the key capability expert users need (Phase 2 in ERT roadmap)
- Phase 1 alone (global settings) does NOT deliver this flexibility

**4. Per-Zone / Per-Region Localization:**
- Right panel allows restricting which geological zones (Zone 1–9) an observation is allowed to update
- Additional dimension beyond ERT's current roadmap
- ERT currently has no zone-based localization concept

**5. Custom Localization Property:**
- Option to supply a **user-defined grid property** as the localization mask
- Gives full flexibility without being limited to parametric distance decay
- ERT's `DistanceLocalizationUpdate` only uses observation-to-parameter distance

**6. Turn Individual Observations Off:**
- Observations can be **turned off** via "No impact" option
- ERT has no per-observation enable/disable in config
- Would require new `ObservationLocalizationConfig` (see Section 8)

### ERT vs ResX Comparison

| Feature | ResX | ERT Current | ERT Update Strategy Setup Goal |
|---------|------|-------------|----------------------------|
| **Per-parameter algorithm** | ✅ Yes | ❌ No (global per experiment) | ✅ Yes (ES/ES-MDA/EnIF per parameter) |
| **Per-observation radius** | ✅ Yes | ❌ No (from `CircleShapeConfig.radius`) | ✅ Yes (editable `main_range`) |
| **Per-zone localization** | ✅ Yes | ❌ No | 🔲 Future (Phase 5?) |
| **Custom localization mask** | ✅ Yes | ❌ No (`DistanceLocalizationUpdate` only) | 🔲 Future |
| **Disable observations** | ✅ Yes ("No impact") | ❌ No | ✅ Yes (`enabled: bool`) |
| **Parameter types** | Grid/Surface/Scalar | Field/Surface/GenKw/EverestControl | ✅ Same + algorithm choice |
| **Algorithm choices** | Adaptive only? | ES/ES-MDA/EnIF + variants | ✅ All + per-parameter |
| **Independent param radii** | ✅ Yes | ❌ No (global localization) | ✅ Yes (Phase 2) |

### ERT's Additional Complexity

ERT has **more algorithm choices** than ResX:
- **ES** (Ensemble Smoother) - single iteration
- **ES-MDA** (Multiple Data Assimilation) - multiple iterations with weights
- **EnIF** (Ensemble Information Filter) - graph-based, experimental

While ResX uses adaptive localization for scalar parameters, ERT can use both:
- **Adaptive localization** (`AdaptiveLocalizationUpdate`) for scalars
- **EnIF algorithm** (graph-based precision matrices) for scalars

This requires the Update Strategy Setup to handle algorithm selection AND localization configuration together.

---

## 1. Project Structure

```
/Users/FCUR/git/ert/src/ert/
├── analysis/              # History matching algorithms and update strategies
│   ├── _es_update.py              # ES/ES-MDA update logic
│   ├── _enif_update.py            # EnIF update logic
│   ├── _update_strategies/        # Strategy pattern implementations
│   │   ├── _global.py             # GlobalESUpdate (no localization)
│   │   ├── _adaptive.py           # AdaptiveLocalizationUpdate
│   │   ├── _distance.py           # DistanceLocalizationUpdate
│   │   └── _protocol.py           # UpdateStrategy protocol
│   └── analysis_module.py         # ESSettings (algorithm configuration)
├── config/                # Configuration classes
│   ├── analysis_module.py         # ESSettings class
│   ├── analysis_config.py         # AnalysisConfig
│   ├── parameter_config.py        # ParameterConfig base class
│   ├── field.py                   # Field parameter
│   ├── surface_config.py          # SurfaceConfig parameter
│   ├── gen_kw_config.py           # GenKwConfig parameter
│   └── _observations.py           # Observation configuration
├── gui/                   # PyQt6-based GUI
│   ├── experiments/               # Experiment configuration panels
│   │   ├── experiment_panel.py            # Main experiment panel (switches between types)
│   │   ├── ensemble_smoother_panel.py    # ES configuration
│   │   ├── multiple_data_assimilation_panel.py  # ES-MDA configuration
│   │   ├── ensemble_information_filter_panel.py # EnIF configuration
│   │   └── manual_update_panel.py        # Manual update panel
│   └── ertwidgets/
│       ├── analysismodulevariablespanel.py  # Current algorithm settings editor
│       └── analysismoduleedit.py           # Settings widget with "Edit" button
└── run_models/            # Experiment run models
    ├── ensemble_smoother.py                # ES run model
    ├── multiple_data_assimilation.py       # ES-MDA run model
    ├── ensemble_information_filter.py      # EnIF run model
    └── update_run_model.py                 # Base update orchestration
```

---

## 2. History Matching Algorithms

### Currently Supported Algorithms

| Algorithm | Implementation | Run Model | Description | Works With Parameter Types |
|-----------|-----------------|-----------|-------------|---------------------------|
| **ES** (Ensemble Smoother) | `_es_update.py:smoother_update()` | `EnsembleSmoother` | Single-iteration ensemble smoother | All parameter types (Field, Surface, GenKw, EverestControl) |
| **ES-MDA** (Ensemble Smoother with Multiple Data Assimilation) | `_es_update.py:smoother_update()` | `MultipleDataAssimilation` | Multiple iteration smoother with configurable weights | All parameter types (Field, Surface, GenKw, EverestControl) |
| **EnIF** (Ensemble Information Filter) | `_enif_update.py:enif_update()` | `EnsembleInformationFilter` | Graph-based parameter update (experimental) | All parameter types (requires `load_parameter_graph()`) |

### Algorithm Compatibility by Parameter Type

Based on the codebase investigation:

| Parameter Type | Dimensionality | Adaptive Localization | Distance Localization | EnIF Support | Graph Structure |
|---------------|----------------|----------------------|---------------------|---------------|----------------|
| **Field** | 3D | ✅ Yes | ✅ Yes (via `DistanceLocalizationUpdate`) | ✅ Yes | 3D grid graph (`create_flattened_cube_graph`) |
| **Surface** | 2D | ✅ Yes | ✅ Yes (via `DistanceLocalizationUpdate`) | ✅ Yes | 2D grid graph (`create_flattened_cube_graph` with pz=1) |
| **GenKw** | 1D (scalar) | ✅ Yes | ❌ No (not spatial) | ✅ Yes | Independent nodes (no edges) |
| **EverestControl** | 1D (scalar) | ✅ Yes | ❌ No | ✅ Yes | Independent nodes (no edges) |

### Key Insight: Parameter Dimensionality

Parameters declare their dimensionality via the `dimensionality` field:
- `Field.dimensionality = 3` (`field.py:42`)
- `SurfaceConfig.dimensionality = 2` (`surface_config.py:62`)
- `GenKwConfig.dimensionality = 1` (`gen_kw_config.py:103`)
- `EverestControl.dimensionality = 1` (`everest_control.py:150`)

This maps directly to ResX's categorization: **grid (3D)**, **surface (2D)**, and **scalar (1D)** parameters.

### Algorithm Settings (`ESSettings` in `config/analysis_module.py`)

```python
class ESSettings(BaseModel):
    enkf_truncation: float = 1.0           # Singular value truncation (0.0-1.0]
    inversion: str = "EXACT"                # Deprecated: "EXACT" or "SUBSPACE"
    localization: bool = False               # Enable adaptive localization
    localization_correlation_threshold: float | None = None  # Custom threshold
    distance_localization: bool = False      # Enable distance-based localization
```

---

## 3. Localization Strategies

### Strategy Pattern (`analysis/_update_strategies/`)

The codebase uses a **strategy pattern** to handle different localization approaches:

| Strategy | File | Description | Used For | Algorithm Base |
|----------|------|-------------|----------|---------------|
| `GlobalESUpdate` | `_global.py` | No localization, uses `iterative_ensemble_smoother.ESMDA` | All parameters when no localization | ES / ES-MDA |
| `AdaptiveLocalizationUpdate` | `_adaptive.py` | Correlation-based thresholding, uses `AdaptiveESMDA` | All parameters when `localization=True` | ES / ES-MDA with adaptive localization |
| `DistanceLocalizationUpdate` | `_distance.py` | Spatial distance-based, uses `LocalizedESMDA` | Field/Surface when `distance_localization=True` | ES / ES-MDA with distance localization |

**Note**: EnIF uses a completely separate code path (`_enif_update.py`) and does NOT use the strategy pattern. It has its own localization via graph-based precision matrices.

### How Strategies Are Selected (`build_strategy_map()` in `_es_update.py:256-346`)

```python
def build_strategy_map(parameters, param_configs, enkf_truncation,
                       distance_localization=False, localization=False, ...):
    if distance_localization:
        # Field/Surface -> DistanceLocalizationUpdate
        # Others -> GlobalESUpdate
    elif localization:
        # All parameters -> AdaptiveLocalizationUpdate
    else:
        # All parameters -> GlobalESUpdate
```

**Limitation**: Currently, the strategy is selected **globally** for all parameters. There is no per-parameter algorithm selection.

---

## 4. Parameter Configuration

### Parameters vs Observations/Responses

**Important**: In ERT, there is a clear distinction:
- **Parameters**: Things to update (Field, Surface, GenKw, EverestControl) - inherit from `ParameterConfig`
- **Observations/Responses**: Data used for history matching (Summary, GenData, RFT, etc.) - inherit from `ResponseConfig` or `Observation` classes

`GenDataConfig` is **NOT** a parameter type - it is a **Response/Observation** type (inherits from `ResponseConfig`).

### Parameter Types (all inherit from `ParameterConfig`)

| Parameter Type | Config Class | Update Default | Dimensionality | Distance Localization | EnIF Graph Structure |
|----------------|--------------|----------------|----------------|----------------------|----------------------|
| **Field** | `Field` (`field.py`) | `update: bool` from config | 3D | ✅ Yes (requires grid: nx, ny, nz, xinc, yinc, origin, rotation) | 3D grid graph via `create_flattened_cube_graph(px, py, pz)` |
| **Surface** | `SurfaceConfig` (`surface_config.py`) | `update: bool` from config | 2D | ✅ Yes (requires: ncol, nrow, xinc, yinc, xori, yori, rotation) | 2D grid graph via `create_flattened_cube_graph(px, py, pz=1)` |
| **GenKw** | `GenKwConfig` (`gen_kw_config.py`) | `update: True` | 1D (scalar) | ❌ No | Independent nodes (no edges) - `nx.Graph()` with single node |
| **EverestControl** | `EverestControl` (`everest_control.py`) | `update: False` | 1D (scalar) | ❌ No | Independent nodes |

### Response/Observation Types (all inherit from `ResponseConfig` or `_observations.py`)

| Response Type | Config Class | Has Location? | Used For Localization? |
|---------------|--------------|---------------|----------------------|
| **SummaryObservation** | `SummaryConfig` + `SummaryObservation` | Optional (via `shape_id`) | ✅ Yes (if location set) |
| **GeneralObservation** | `GeneralObservation` | Optional (via `shape_id`) | ✅ Yes (if location set) |
| **GenData** | `GenDataConfig` (`gen_data_config.py`) | ❌ No | ❌ No |
| **RFTObservation** | `RFTObservation` | ✅ Yes (required) | ✅ Yes (always has east, north) |
| **DerivedResponse** | `DerivedResponseConfig` | ❌ No | ❌ No |

### ParameterConfig Base Class (`parameter_config.py`)

```python
class ParameterConfig(BaseModel):
    type: str           # "field", "surface", "gen_kw", etc.
    name: str
    forward_init: bool
    update: bool        # Whether this parameter should be updated

    @abstractmethod
    def parameter_keys(self) -> list[str]:
        """Returns a list of parameter keys within this parameter group"""

    @abstractmethod
    def load_parameter_graph(self) -> nx.Graph[int]:
        """Load the graph encoding Markov properties (for EnIF)"""

    @property
    def dimensionality(self) -> Literal[1, 2, 3]:
        """Returns 1, 2, or 3 based on parameter type"""
```

### Key Difference: EnIF vs ES/ES-MDA

- **ES/ES-MDA** (in `_es_update.py`): Uses the **strategy pattern** (`GlobalESUpdate`, `AdaptiveLocalizationUpdate`, `DistanceLocalizationUpdate`)
- **EnIF** (in `_enif_update.py`): Uses a **completely separate code path**:
  - Does NOT use the strategy pattern
  - Uses `graphite_maps.enif.EnIF` for graph-based updates
  - Localization is achieved via **precision matrices** (`Prec_u` for parameters, `Prec_eps` for observations)
  - Requires `load_parameter_graph()` to return a `nx.Graph[int]` for each parameter group
  - GenKw parameters return a graph with no edges (independent parameters)

---

## 5. Observation Configuration & Localization

### Observations/Responses (NOT Parameters!)

**Important Distinction**: Observations are data used for history matching. They are:
- Configured via `GEN_DATA`, `SUMMARY_OBSERVATION`, `GENERAL_OBSERVATION`, `RFT_OBSERVATION` keywords
- Stored as `ResponseConfig` subclasses in `EnsembleConfig.response_configs`
- Includ`GenDataConfig`, `SummaryConfig`, `RFTConfig`, `GeneralObservation`, `SummaryObservation`

**GenDataConfig is a Response/Observation type** - NOT a parameter!

### Observation Types with Localization Support (`_observations.py`)

| Observation Type | Config Class | Has Location? | Location Fields | Localization Support |
|------------------|--------------|---------------|-----------------|---------------------|
| **SummaryObservation** | `SummaryObservation` (`_observations.py`) | Optional | `shape_id` -> `CircleShapeConfig` (east, north, radius) | ✅ Yes (if shape_id set) |
| **GeneralObservation** | `GeneralObservation` (`_observations.py`) | Optional | `shape_id` -> `CircleShapeConfig` | ✅ Yes (if shape_id set) |
| **RFTObservation** | `RFTObservation` (`_observations.py`) | Yes (required) | Direct: `east`, `north`, `tvd`, `md` + `shape_id` | ✅ Yes (always) |

### Response Types (Data only, no localization)

| Response Type | Config Class | Used For |
|---------------|--------------|----------|
| **GenData** | `GenDataConfig` (`gen_data_config.py`) | General data responses (NO location data) |
| **Summary** | `SummaryConfig` (`summary_config.py`) | Summary key responses (location optional via observation) |
| **RFT** | `RFTConfig` (`rft_config.py`) | RFT responses (has location) |
| **DerivedResponse** | `DerivedResponseConfig` (`derived_response_config.py`) | Computed responses |

### Observation Location Data Structures

**`CircleShapeConfig` (`_shapes.py`)** - Used for distance-based localization:
```python
class CircleShapeConfig(ShapeConfig):
    type: Literal["circle"] = "circle"
    east: float      # X-coordinate (meters)
    north: float     # Y-coordinate (meters)
    radius: float    # Localization radius (meters) - maps to `main_range`
```

**`ObservationLocations` (`_update_strategies/_protocol.py`)** - Passed to `DistanceLocalizationUpdate.prepare()`:
```python
@dataclass
class ObservationLocations:
    xpos: npt.NDArray          # X coordinates of observations
    ypos: npt.NDArray          # Y coordinates of observations
    main_range: npt.NDArray    # Correlation range for each observation (from radius)
    location_mask: npt.NDArray   # Boolean mask for valid location data
```

### How Distance-Based Localization Works (in `_distance.py`)

1. Observation locations are extracted (xpos, ypos, main_range)
2. For each Field/Surface parameter, compute `rho_matrix` based on distance:
   - `calc_rho_for_2d_grid_layer()` computes spatial correlation
   - Uses observation coordinates and parameter grid definition
3. The `localization_callback` applies the correlation factor to the Kalman gain

**Current Limitation**: The `main_range` (correlation range) comes from `CircleShapeConfig.radius`, but there's no per-observation customization in the GUI.

### Mapping ResX Concept to ERT

| ResX Concept | ERT Equivalent |
|--------------|----------------|
| Grid parameter localization settings | `Field` parameters with `DistanceLocalizationUpdate` (3D) |
| Surface parameter localization settings | `SurfaceConfig` parameters with `DistanceLocalizationUpdate` (2D) |
| Scalar parameter localization settings | `GenKw`/`EverestControl` parameters with `AdaptiveLocalizationUpdate` OR use `EnIF` algorithm |
| Per-observation correlation range | `CircleShapeConfig.radius` -> `ObservationLocations.main_range` |

---

## 6. Current GUI Structure

### Experiment Panel Flow (`gui/experiments/experiment_panel.py`)

1. User selects experiment type from combo box:
   - Single Test Run
   - Ensemble Experiment
   - Evaluate Ensemble
   - Multiple Data Assimilation (ES-MDA)
   - Ensemble Smoother (ES)
   - Ensemble Information Filter (EnIF)
   - Manual Update

2. Each experiment type has its own configuration panel

### Current Algorithm Settings UI (`analysismodulevariablespanel.py`)

The `AnalysisModuleVariablesPanel` provides a simple UI for:
- Singular value truncation (spinbox)
- Adaptive localization checkbox
- Correlation threshold spinner (enabled when localization checked)

**Limitations**:
- Settings are **per-experiment**, not per-parameter
- No way to mix algorithms (e.g., use ES-MDA for some parameters, EnIF for others)
- No per-observation localization settings
- Distance-based localization is a global toggle

---

## 7. Key Findings for New UI Design

### What Exists
1. ✅ Strategy pattern allows different update algorithms per parameter (via `build_strategy_map()`)
2. ✅ Parameters have `update: bool` to enable/disable updates
3. ✅ Parameters declare `dimensionality` (1D, 2D, 3D) for categorization
4. ✅ Clear separation between **Parameters** (`ParameterConfig`) and **Observations/Responses** (`ResponseConfig`/`Observation`)
5. ✅ Observations can have location data (east, north, radius) via `CircleShapeConfig`
6. ✅ `DistanceLocalizationUpdate` handles **Field/Surface parameters** with spatial localization
7. ✅ `ObservationLocations` dataclass holds per-observation location data (xpos, ypos, main_range)
8. ✅ EnIF has its own graph-based localization via `load_parameter_graph()` and precision matrices
9. ✅ `GenKwConfig.load_parameter_graph()` returns independent nodes (no edges) for scalar parameters

### What's Missing for the New UI
1. ❌ **Per-parameter algorithm selection**: Currently, `build_strategy_map()` applies the same algorithm to all parameters (with only distance vs. adaptive vs. global distinction)
2. ❌ **Per-observation localization properties**: No GUI to set/modify localization radius per observation (currently uses `CircleShapeConfig.radius`)
3. ❌ **Algorithm-specific settings per parameter**: No way to use different truncation/correlation settings for different parameters
4. ❌ **Mixing algorithms in same run**: Currently, you choose ONE algorithm per experiment run
5. ❌ **Unified Update Strategy Setup panel**: Need to replace separate ES/ES-MDA/EnIF panels with a single panel like ResX
6. ❌ **EnIF integration with strategy pattern**: EnIF uses completely separate code path (`_enif_update.py`)

### Data Model Gaps to Address
1. Need to extend `build_strategy_map()` to accept per-parameter strategy specifications (or create new unified update function)
2. Need to store per-observation localization settings (currently `main_range` is derived from `CircleShapeConfig.radius`)
3. Need to support mixing algorithms (e.g., ES-MDA for some parameters, EnIF for others in the same run)
4. Need to handle EnIF's unique requirements (graph-based precision matrices) within the unified design

---

## 8. Proposed Data Model Changes

### New: Per-Parameter Update Configuration

```python
class ParameterUpdateConfig(BaseModel):
    parameter_name: str
    algorithm: Literal["ES", "ES-MDA", "EnIF"] = "ES-MDA"
    update: bool = True

    # Parameter type categorization (derived from parameter.type/dimensionality)
    parameter_type: Literal["grid", "surface", "scalar"]  # Derived from dimensionality

    # Algorithm-specific settings
    enkf_truncation: float | None = None  # None = use default
    localization_type: Literal["none", "adaptive", "distance"] = "none"
    localization_correlation_threshold: float | None = None  # For adaptive

    # Distance localization settings (for grid/surface parameters)
    use_distance_localization: bool = False
```

### New: Per-Observation Localization Settings

```python
class ObservationLocalizationConfig(BaseModel):
    observation_name: str
    east: float | None = None      # Override from config's CircleShapeConfig
    north: float | None = None     # Override from config's CircleShapeConfig
    main_range: float | None = None  # Correlation range (meters) - overrides radius
    enabled: bool = True            # Whether to use for localization
```

### Modified: Unified Update Strategy Map Builder

Instead of current `build_strategy_map()` which takes global settings, create a new function that accepts per-parameter configurations:

```python
def build_unified_strategy_map(
    parameters: Iterable[str],
    param_configs: Mapping[str, ParameterConfig],
    param_update_configs: Mapping[str, ParameterUpdateConfig],
    obs_localization_configs: Mapping[str, ObservationLocalizationConfig],
    ensemble_size: int,
    progress_callback: Callable[[AnalysisEvent], None],
) -> dict[str, UpdateStrategy | EnIFHandler]:
    """
    Build strategy map supporting mixed algorithms and per-parameter settings.

    For EnIF parameters, returns an EnIFHandler instead of UpdateStrategy.
    """
    pass
```

### Key Design Decision: Handling EnIF

Since EnIF uses a completely different code path (`_enif_update.py`), we have two options:

**Option A**: Keep EnIF separate, don't allow mixing with ES/ES-MDA in same run
- Simpler to implement
- Matches current behavior
- Limitation: Can't use EnIF for some params and ES-MDA for others

**Option B**: Integrate EnIF into a unified framework
- Requires significant refactoring of `_enif_update.py`
- Would need to create an `EnIFUpdateStrategy` that implements `UpdateStrategy` protocol
- Benefit: True per-parameter algorithm selection

**Recommendation**: Start with **Option A** (EnIF as separate algorithm choice for ALL parameters), then evolve to Option B in future.

---

## 9. Recommended UI Components (ResX-Inspired Design)

### Update Strategy Setup Panel (New) - Main Tab Structure

A new **"Update Strategy Setup"** tab/panel that replaces the current experiment-type-specific panels with a unified design inspired by ResX's **"Define localization setup"** tab. The Svelte prototype implements the following tab structure:

#### Tab 1: Parameters (Algorithm Selection)
- Uses `AlgorithmSelection` component
- **Parameter table** with columns:
  - Checkbox: "Update" (from `ParameterConfig.update`)
  - Parameter name
  - Parameter type (Grid 3D / Surface 2D / Scalar 1D) - derived from `dimensionality`
  - Dropdown: Algorithm (ES / ES-MDA / EnIF)
  - Button: "Settings..." (opens algorithm-specific dialog)

#### Tab 2: Observations (Observation Localization)
- Uses `ObservationLocalization` component
- Table with all **observations that have location data** (have `shape_id` or are `RFTObservation`)
- Sources: `EnsembleConfig.response_configs` (observations with location from `CircleShapeConfig`)
- Columns:
  - Observation name (from `SummaryObservation`, `GeneralObservation`, `RFTObservation`)
  - Type (Summary/General/RFT)
  - X position (east)
  - Y position (north)
  - **Main range (correlation range)** - editable per-observation
  - Enabled (checkbox) - whether to use for localization
- Visual indicator: which parameters this observation affects

#### Tab 3: Parameters vs. Observations (Matrix View)
- Uses `ParameterObservationMatrix` component
- Matrix view showing relationships between parameters and observations
- Visualizes parameter-observation correlations or update mappings

#### Tab 4: Summary/Validation
- Uses `SummaryValidation` component
- Shows which parameters use which algorithms
- Shows which observations are used for which parameters
- Validation warnings (e.g., observation without location for distance localization)
- "Run" button to start the experiment

Note: The originally proposed "Localization Setup" tab (with Grid/Surface/Scalar subsections) is not included in the current prototype.

---

## 10. File Paths for Implementation

### Files to Modify
**Core logic:**
- `src/ert/config/analysis_module.py` - Extend `ESSettings` or create new config class for per-parameter settings
- `src/ert/analysis/_es_update.py` - Modify `build_strategy_map()` to accept per-parameter configs
- `src/ert/analysis/_enif_update.py` - Integrate with new per-parameter design (or keep separate)
- `src/ert/analysis/_update_strategies/_protocol.py` - May need to extend `UpdateStrategy` for EnIF compatibility

**Config classes:**
- `src/ert/config/parameter_config.py` - ParameterConfig base class (for Parameters only)
- `src/ert/config/response_config.py` - ResponseConfig base class (for Observations/Responses)
- `src/ert/config/gen_data_config.py` - GenDataConfig is a **Response**, NOT a Parameter!
- `src/ert/config/_observations.py` - Observation classes (SummaryObservation, GeneralObservation, RFTObservation)

**GUI:**
- `src/ert/gui/experiments/experiment_panel.py` - Replace experiment-type combo with new Update Strategy Setup
- `src/ert/gui/experiments/ensemble_smoother_panel.py` - Refactor into new design
- `src/ert/gui/experiments/multiple_data_assimilation_panel.py` - Refactor into new design
- `src/ert/gui/experiments/ensemble_information_filter_panel.py` - Integrate into new design
- `src/ert/gui/ertwidgets/analysismodulevariablespanel.py` - Replace with new widgets

### Files to Create
**New config classes:**
- `src/ert/config/update_designer_config.py` - New config classes (`ParameterUpdateConfig`, `ObservationLocalizationConfig`)

**New GUI components:**
- `src/ert/gui/experiments/update_designer_panel.py` - Main new panel (replaces experiment-type sub-panels)
- `src/ert/gui/ertwidgets/parameter_update_table.py` - Parameter configuration table with algorithm selection
- `src/ert/gui/ertwidgets/localization_setup_widget.py` - ResX-inspired tab for grid/surface/scalar localization
- `src/ert/gui/ertwidgets/observation_localization_table.py` - Observation localization table (for observations WITH location data)
- `src/ert/gui/ertwidgets/algorithm_settings_dialog.py` - Per-algorithm settings dialog

### Key: Parameters vs Observations

Remember the distinction when implementing:
- **Parameters** (update targets): `Field`, `SurfaceConfig`, `GenKwConfig`, `EverestControl` - all inherit from `ParameterConfig`
- **Observations/Responses** (data for history matching): `GenDataConfig`, `SummaryConfig`, `RFTConfig`, `SummaryObservation`, `GeneralObservation`, `RFTObservation` - inherit from `ResponseConfig` or observation classes in `_observations.py`

### EnIF Integration Consideration
Since EnIF uses a completely separate code path (`_enif_update.py`), you have two implementation approaches:

**Approach 1: Keep EnIF Separate (Simpler, Recommended for v1)**
- Update Strategy Setup allows choosing EnIF as a "global" algorithm (all parameters)
- ES/ES-MDA settings and EnIF settings are in separate sections
- Don't allow mixing EnIF with ES/ES-MDA in same run

**Approach 2: Full Integration (Complex, Future)**
- Create `EnIFUpdateStrategy` that implements `UpdateStrategy` protocol
- Refactor `_enif_update.py` to use the same `perform_ensemble_update()` function
- Allow per-parameter algorithm selection including EnIF

---

## 11. References

| Description | File Path |
|-------------|-----------|
| Strategy pattern protocol | `src/ert/analysis/_update_strategies/_protocol.py` |
| ES/ES-MDA update logic | `src/ert/analysis/_es_update.py` |
| EnIF update logic | `src/ert/analysis/_enif_update.py` |
| Distance localization | `src/ert/analysis/_update_strategies/_distance.py` |
| Algorithm settings | `src/ert/config/analysis_module.py` |
| Parameter base class | `src/ert/config/parameter_config.py` |
| Field parameter | `src/ert/config/field.py` |
| Surface parameter | `src/ert/config/surface_config.py` |
| Observation config | `src/ert/config/_observations.py` |
| Shape config (for localization) | `src/ert/config/_shapes.py` |
| Current algorithm UI | `src/ert/gui/ertwidgets/analysismodulevariablespanel.py` |
| Current experiment panel | `src/ert/gui/experiments/experiment_panel.py` |
