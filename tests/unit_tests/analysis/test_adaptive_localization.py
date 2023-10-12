from argparse import ArgumentParser

import numpy as np
import pytest

from ert import LibresFacade
from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.storage import open_storage


def run_cli_ES_with_case(ert_config):
    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "prior_sample",
            "--target-case",
            "posterior_sample",
            "--realizations",
            "1-50",
            ert_config,
            "--port-range",
            "1024-65535",
        ],
    )

    run_cli(parsed)
    facade = LibresFacade.from_config_file(ert_config)
    with open_storage(facade.enspath) as storage:
        prior_ensemble_name = storage.get_ensemble_by_name("prior_sample")
        prior_sample = facade.load_all_gen_kw_data(prior_ensemble_name)
        posterior_ensemble_name = storage.get_ensemble_by_name("posterior_sample")
        posterior_sample = facade.load_all_gen_kw_data(posterior_ensemble_name)

    return prior_sample, posterior_sample


@pytest.mark.integration_test
def test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior(copy_case):
    copy_case("poly_example")

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

    prior_sample_loc0, posterior_sample_loc0 = run_cli_ES_with_case("poly_loc_0.ert")
    prior_sample_noloc, posterior_sample_noloc = run_cli_ES_with_case("poly_no_loc.ert")

    # Check posterior sample without adaptive localization and with cut-off 0 are equal
    assert np.array_equal(posterior_sample_loc0, posterior_sample_noloc)

