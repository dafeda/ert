from argparse import ArgumentParser

import numpy as np

from ert import LibresFacade
from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.storage import open_storage


def test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior(copy_case):
    copy_case("poly_example")

    parser = ArgumentParser(prog="test_main")
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "default",
            "--target-case",
            "target",
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
        default_fs = storage.get_ensemble_by_name("default")
        df_default = facade.load_all_gen_kw_data(default_fs)
        target_fs = storage.get_ensemble_by_name("target")
        df_target = facade.load_all_gen_kw_data(target_fs)

    # We expect that when cut-off is 1, the posterior is equal to prior
    assert (
        np.linalg.det(df_target.cov().to_numpy()) == np.linalg.det(df_default.cov().to_numpy())
    )


def test_that_adaptive_localization_with_cutoff_0_equals_ESupdate(copy_case):
    copy_case("poly_example")

    parser = ArgumentParser(prog="test_main_0")
    parsed_loc0 = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "default_loc0",
            "--target-case",
            "target_loc0",
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
        target_fs_loc0 = storage_loc0.get_ensemble_by_name("target_loc0")
        posterior_loc0 = facade_loc0.load_all_gen_kw_data(target_fs_loc0)

    parser_noloc = ArgumentParser(prog="test_main")
    parsed_noloc = ert_parser(
        parser_noloc,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "default_noloc",
            "--target-case",
            "target_noloc",
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
        target_fs_noloc = storage_noloc.get_ensemble_by_name("target_noloc")
        posterior_noloc = facade_noloc.load_all_gen_kw_data(target_fs_noloc)

    # We expect that when cut-off is 0, the posterior is equal to ES
    assert (
        np.linalg.det(posterior_loc0.cov().to_numpy()) == np.linalg.det(posterior_noloc.cov().to_numpy())
    )
