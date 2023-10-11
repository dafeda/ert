from argparse import ArgumentParser

import numpy as np
import pytest

from ert import LibresFacade
from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.storage import open_storage


@pytest.mark.integration_test
def test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior(copy_case):
    copy_case("poly_example")

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
            "poly_loc_1.ert",
            "--port-range",
            "1024-65535",
        ],
    )

    run_cli(parsed)
    facade = LibresFacade.from_config_file("poly_loc_1.ert")
    with open_storage(facade.enspath) as storage:
        prior_ensemble_name = storage.get_ensemble_by_name("prior_sample")
        prior_sample = facade.load_all_gen_kw_data(prior_ensemble_name)
        posterior_ensemble_name = storage.get_ensemble_by_name("posterior_sample")
        posterior_sample = facade.load_all_gen_kw_data(posterior_ensemble_name)

    # Check prior and posterior samples are equal
    assert np.array_equal(posterior_sample, prior_sample)


@pytest.mark.integration_test
def test_that_adaptive_localization_with_cutoff_0_equals_ESupdate(copy_case):
    """
    Note that "RANDOM_SEED" in poly_loc_0.ert and poly_no_loc.ert needs to be the same to obtain 
    the same sample from the prior.
    """
    copy_case("poly_example")

    parser = ArgumentParser(prog="test_main_0")
    parsed_loc0 = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "prior_sample_loc0",
            "--target-case",
            "posterior_sample_loc0",
            "--realizations",
            "1-50",
            "poly_loc_0.ert",
            "--port-range",
            "1024-65535",
        ],
    )

    run_cli(parsed_loc0)
    facade_loc0 = LibresFacade.from_config_file("poly_loc_0.ert")
    with open_storage(facade_loc0.enspath) as storage_loc0:
        posterior_ensemble_name_loc0 = storage_loc0.get_ensemble_by_name("posterior_sample_loc0")
        posterior_sample_loc0 = facade_loc0.load_all_gen_kw_data(posterior_ensemble_name_loc0)

    parser_noloc = ArgumentParser(prog="test_main")
    parsed_noloc = ert_parser(
        parser_noloc,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "prior_sample_noloc",
            "--target-case",
            "posterior_sample_noloc",
            "--realizations",
            "1-50",
            "poly_no_loc.ert",
            "--port-range",
            "1024-65535",
        ],
    )

    run_cli(parsed_noloc)
    facade_noloc = LibresFacade.from_config_file("poly.ert")
    with open_storage(facade_noloc.enspath) as storage_noloc:
        posterior_ensemble_name_noloc = storage_noloc.get_ensemble_by_name("posterior_sample_noloc")
        posterior_sample_noloc = facade_noloc.load_all_gen_kw_data(posterior_ensemble_name_noloc)

    # Check posterior sample without adaptive localization and with cut-off 0 are equal
    assert np.array_equal(posterior_sample_loc0, posterior_sample_noloc)

