from argparse import ArgumentParser
from textwrap import dedent

import numpy as np
import pytest

from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.config import ErtConfig
from ert.storage import open_storage


def run_cli_ES_with_case(poly_config):
    config_name = poly_config.split(".")[0]
    prior_sample_name = "prior_sample"+"_"+config_name
    posterior_sample_name = "posterior_sample"+"_"+config_name
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            prior_sample_name,
            "--target-case",
            posterior_sample_name,
            "--realizations",
            "1-50",
            poly_config,
            "--port-range",
            "1024-65535",
        ],
    )
    run_cli(parsed)
    storage_path = ErtConfig.from_file(poly_config).ens_path
    #facade = LibresFacade.from_config_file(ert_config)
    with open_storage(storage_path) as storage:
        prior_ensemble = storage.get_ensemble_by_name(prior_sample_name)
        prior_sample = prior_ensemble.load_parameters("COEFFS")
        posterior_ensemble = storage.get_ensemble_by_name(posterior_sample_name)
        posterior_sample = posterior_ensemble.load_parameters("COEFFS")
    return prior_sample, posterior_sample


@pytest.mark.integration_test
def test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior(copy_case):
    copy_case("poly_example")
    random_seed_line = "RANDOM_SEED 1234\n\n"
    set_adaptive_localization_1 = dedent(
        """
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 1.0
        """
    )

    with open("poly.ert", "r+") as f:
        lines = f.readlines()
        lines.insert(2, random_seed_line)
        lines.insert(10, set_adaptive_localization_1)

    with open("poly_loc_1.ert", "w") as f:
        f.writelines(lines)
    prior_sample, posterior_sample = run_cli_ES_with_case("poly_loc_1.ert")

    # Check prior and posterior samples are equal
    assert np.array_equal(posterior_sample, prior_sample)


@pytest.mark.integration_test
def test_that_adaptive_localization_with_cutoff_0_equals_ESupdate(copy_case):
    """
    Note that "RANDOM_SEED" in poly_loc_0.ert and poly_no_loc.ert needs to be the same to obtain 
    the same sample from the prior.
    """
    copy_case("poly_example")

    random_seed_line = "RANDOM_SEED 1234\n\n"
    set_adaptive_localization_0 = dedent(
        """
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION True
        ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD 0.0
        """
    )

    with open("poly.ert", "r+") as f:
        lines = f.readlines()
        lines.insert(2, random_seed_line)

    with open("poly_no_loc.ert", "w") as f:
        f.writelines(lines)

    lines.insert(10, set_adaptive_localization_0)

    with open("poly_loc_0.ert", "w") as f:
        f.writelines(lines)

    prior_sample_loc0, posterior_sample_loc0 = run_cli_ES_with_case("poly_loc_0.ert")
    prior_sample_noloc, posterior_sample_noloc = run_cli_ES_with_case("poly_no_loc.ert")

    # Check posterior sample without adaptive localization and with cut-off 0 are equal
    assert np.array_equal(posterior_sample_loc0, posterior_sample_noloc)

