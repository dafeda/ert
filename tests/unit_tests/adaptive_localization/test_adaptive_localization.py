from argparse import Namespace
from uuid import UUID

from ert.cli import model_factory


def test_that_adaptive_localization_with_cutoff_1_equals_ensemble_prior(poly_case, storage):
    #ert = poly_loc_1_case
    ert = poly_case
    experiment_id = storage.create_experiment(
        parameters=ert.ensembleConfig().parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=5
    )
    prior = ert.ensemble_context(prior_ensemble, [True, False, False, True, True], 0)
    ert.sample_prior(prior.sim_fs, prior.active_realizations)
    ens_config = ert.ensembleConfig()
    args = Namespace(realizations=None, iter_num=1, current_case="default")
    model = model_factory._setup_ensemble_experiment(
        ert,
        storage,
        args,
        UUID(int=0),
    )
    i = 10


