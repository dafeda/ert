from pathlib import Path

import pytest

from ert.config import (
    AnalysisParameterType,
    ErtConfig,
    ParameterUpdateStrategy,
)
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage
from tests.ert.ui_tests.cli.run_cli import run_cli


def assert_variance_in_field(
    prior_ensemble, posterior_ensemble, field_name, obs_pos, no_upd_pos
):
    prior_data = prior_ensemble.load_parameters(field_name)
    posterior_data = posterior_ensemble.load_parameters(field_name)

    prior_var_obs = prior_data.var(dim="realizations", ddof=1).sel(
        x=obs_pos[0], y=obs_pos[1]
    )
    posterior_var_obs = posterior_data.var(dim="realizations", ddof=1).sel(
        x=obs_pos[0], y=obs_pos[1]
    )

    prior_var_no_obs = prior_data.var(dim="realizations", ddof=1).sel(
        x=no_upd_pos[0], y=no_upd_pos[1]
    )
    posterior_var_no_obs = posterior_data.var(dim="realizations", ddof=1).sel(
        x=no_upd_pos[0], y=no_upd_pos[1]
    )

    obs_reduction = prior_var_obs.mean() - posterior_var_obs.mean()
    no_obs_reduction = prior_var_no_obs.mean() - posterior_var_no_obs.mean()

    assert obs_reduction > no_obs_reduction, (
        f"Expecting stronger variance reduction at observation location "
        f"than outside on {field_name}"
    )


@pytest.mark.timeout(600)
@pytest.mark.usefixtures("copy_heat_equation")
@pytest.mark.slow
def test_that_parameter_type_distance_strategy_runs_on_heat_equation():
    with Path("config.ert").open(encoding="utf-8") as fh:
        lines = fh.readlines()

    config_content = [
        line for line in lines if not line.lstrip().startswith("ANALYSIS_SET_VAR")
    ]
    config_content.extend(
        [
            "ANALYSIS_SET_VAR PARAMETERS FIELD DISTANCE\n",
            "ENSPATH heat_storage_type_dl\n",
        ]
    )
    with Path("heat_type_dl.ert").open("w", encoding="utf-8") as fh:
        fh.writelines(config_content)

    run_cli(
        ENSEMBLE_SMOOTHER_MODE,
        "--disable-monitoring",
        "heat_type_dl.ert",
        "--experiment-name",
        "heat_type_dl",
    )

    config = ErtConfig.from_file("heat_type_dl.ert")
    assert config.analysis_config.es_settings.localization is False
    assert config.analysis_config.es_settings.parameter_update_strategies == {
        AnalysisParameterType.FIELD: ParameterUpdateStrategy.DISTANCE,
    }

    with open_storage(config.ens_path) as storage:
        experiment = storage.get_experiment_by_name("heat_type_dl")
        prior = experiment.get_ensemble_by_name("iter-0")
        posterior = experiment.get_ensemble_by_name("iter-1")
        assert_variance_in_field(prior, posterior, "COND", (2, 2), (9, 9))
