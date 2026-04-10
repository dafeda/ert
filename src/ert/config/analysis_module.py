from __future__ import annotations

import logging
import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

logger = logging.getLogger(__name__)


DEFAULT_ENKF_TRUNCATION_EXACT = 1.0
DEFAULT_ENKF_TRUNCATION_SUBSPACE = 0.98
DEFAULT_LOCALIZATION = False


def _upper(v: str) -> str:
    return v.upper()


InversionTypeES = Annotated[Literal["EXACT", "SUBSPACE"], BeforeValidator(_upper)]


class AnalysisParameterType(StrEnum):
    FIELD = "field"
    GEN_KW = "gen_kw"
    SURFACE = "surface"
    EVEREST_PARAMETERS = "everest_parameters"


class ParameterUpdateStrategy(StrEnum):
    STANDARD = "STANDARD"
    ADAPTIVE = "ADAPTIVE"
    DISTANCE = "DISTANCE"


DISTANCE_LOCALIZATION_PARAMETER_TYPES = {
    AnalysisParameterType.FIELD,
    AnalysisParameterType.SURFACE,
}

es_description = """
    Deprecated. Use enkf_truncation instead.
    truncation = 1.0 corresponds to EXACT inversion.
    truncation < 1.0 corresponds to SUBSPACE inversion.
    """

loc_description = """
    The default adaptive localization correlation threshold
    is computed as 3/sqrt(ensemble_size), where ensemble_size
    is the number of active realizations in the ensemble.

    You can override this value by setting a custom threshold here or in the config.
    """

cust_loc_thresh_description = """
    Adaptive localization correlation threshold:
    """


class ESSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    enkf_truncation: Annotated[
        float,
        Field(gt=0.0, le=1.0, title="Singular value truncation"),
    ] = DEFAULT_ENKF_TRUNCATION_EXACT
    inversion: Annotated[
        InversionTypeES, Field(title="Inversion algorithm", description=es_description)
    ] = "EXACT"

    @model_validator(mode="before")
    @classmethod
    def _default_enkf_truncation_from_inversion(cls, data: Any) -> Any:
        if isinstance(data, dict):
            legacy_distance_localization = data.pop("distance_localization", None)
            if str(legacy_distance_localization).strip().lower() == "true":
                if str(data.get("localization", False)).strip().lower() == "true":
                    raise ValueError(
                        "LOCALIZATION and DISTANCE_LOCALIZATION cannot both be enabled"
                    )
                if data.get("parameter_update_strategies"):
                    raise ValueError(
                        "PARAMETERS update strategies cannot be combined with "
                        "DISTANCE_LOCALIZATION"
                    )

                data["parameter_update_strategies"] = {
                    AnalysisParameterType.FIELD: ParameterUpdateStrategy.DISTANCE,
                    AnalysisParameterType.SURFACE: ParameterUpdateStrategy.DISTANCE,
                }

            if "enkf_truncation" not in data:
                inversion = str(data.get("inversion", "EXACT")).upper()
                data["enkf_truncation"] = (
                    DEFAULT_ENKF_TRUNCATION_SUBSPACE
                    if inversion == "SUBSPACE"
                    else DEFAULT_ENKF_TRUNCATION_EXACT
                )
        return data

    localization: Annotated[
        bool, Field(title="Enable adaptive localization", description=loc_description)
    ] = False
    localization_correlation_threshold: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            title="Custom adaptive localization correlation threshold",
            description=cust_loc_thresh_description,
        ),
    ] = None
    parameter_update_strategies: Annotated[
        dict[AnalysisParameterType, ParameterUpdateStrategy],
        Field(title="Parameter-type update strategies"),
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_parameter_update_strategies(self) -> ESSettings:
        invalid_parameter_types = [
            parameter_type.name
            for parameter_type, strategy in self.parameter_update_strategies.items()
            if strategy == ParameterUpdateStrategy.DISTANCE
            and parameter_type not in DISTANCE_LOCALIZATION_PARAMETER_TYPES
        ]
        if invalid_parameter_types:
            raise ValueError(
                "DISTANCE strategy is only supported for FIELD and SURFACE "
                "parameter types"
            )
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_empty_parameter_update_strategies(
        self, handler: Any
    ) -> dict[str, Any]:
        serialized = handler(self)
        if not self.parameter_update_strategies:
            serialized.pop("parameter_update_strategies", None)
        return serialized

    def correlation_threshold(self, ensemble_size: int) -> float:
        """Decides whether to use user-defined or default threshold.

        Default threshold taken from luo2022,
        Continuous Hyper-parameter Optimization (CHOP) in an ensemble Kalman filter
        Section 2.3 - Localization in the CHOP problem
        """
        if self.localization_correlation_threshold is None:
            return 3 / math.sqrt(ensemble_size)
        else:
            return self.localization_correlation_threshold


AnalysisModule = ESSettings
