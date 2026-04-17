from unittest.mock import PropertyMock, patch

import pytest

from ert.config import GenKwConfig
from ert.config.gen_kw_config import DataSource
from ert.sample_prior import sample_prior


def test_that_sample_prior_raises_when_design_matrix_source_has_no_dataframe(
    storage,
):
    param = GenKwConfig(
        name="DM_PARAM",
        distribution={"name": "raw"},
        input_source=DataSource.DESIGN_MATRIX,
        update=False,
    )
    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [
                param.model_dump(mode="json"),
            ],
        },
    )
    ensemble = storage.create_ensemble(experiment, name="prior", ensemble_size=1)

    with pytest.raises(
        ValueError,
        match="uses a design matrix, but no design matrix was provided",
    ):
        sample_prior(ensemble, [0], random_seed=42, num_realizations=1)


def test_that_sample_prior_raises_on_unhandled_input_source(
    storage,
):
    param = GenKwConfig(
        name="MY_PARAM",
        distribution={"name": "normal", "mean": 0, "std": 1},
    )
    experiment = storage.create_experiment(
        experiment_config={
            "parameter_configuration": [param.model_dump(mode="json")],
        },
    )
    ensemble = storage.create_ensemble(experiment, name="prior", ensemble_size=1)

    bogus_param = GenKwConfig(
        name="MY_PARAM",
        distribution={"name": "normal", "mean": 0, "std": 1},
    )
    bogus_param.__dict__["input_source"] = "bogus"
    bogus_config = {"MY_PARAM": bogus_param}

    with (
        patch.object(
            type(experiment),
            "parameter_configuration",
            new_callable=PropertyMock,
            return_value=bogus_config,
        ),
        pytest.raises(
            NotImplementedError,
            match="Unhandled input source for 'MY_PARAM': bogus",
        ),
    ):
        sample_prior(ensemble, [0], random_seed=42, num_realizations=1)
