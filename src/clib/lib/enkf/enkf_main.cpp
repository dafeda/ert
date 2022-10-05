#include <ert/python.hpp>

#include <ert/logging.hpp>

static auto logger = ert::get_logger("enkf");

namespace fs = std::filesystem;

namespace enkf_main {
/**
  @brief Substitutes the sampled parameters into the runpath.

  Handles the substitution of all sampled parameter values into parameter
  templates. E.g. for configs including `GEN_KW key template_file target_file`,
  sampled values are gotten from fs, replaced in the contents of template_file
  and written to target_file in the runpath.

  @param ens_config Where to find the nodes (e.g. `GEN_KW key template_file
    target_file` definition).
  @param export_base_name The base name of the value_export file (e.g. if
    "parameters", value export file will e.g. be "parameters.json")
  @param run_path The run__path to write the target file in.
  @param iens The realization number.
  @param fs The enkf_fs to load sampled parameters from
*/
void ecl_write(const ensemble_config_type *ens_config,
               const char *export_base_name, const char *run_path, int iens,
               enkf_fs_type *fs) {
    value_export_type *export_value =
        value_export_alloc(run_path, export_base_name);

    for (auto &key : ensemble_config_keylist_from_var_type(
             ens_config, PARAMETER + EXT_PARAMETER)) {
        enkf_node_type *enkf_node =
            enkf_node_alloc(ensemble_config_get_node(ens_config, key.c_str()));
        node_id_type node_id = {.report_step = 0, .iens = iens};

        if (enkf_node_use_forward_init(enkf_node) &&
            !enkf_node_has_data(enkf_node, fs, node_id))
            continue;
        enkf_node_load(enkf_node, fs, node_id);

        enkf_node_ecl_write(enkf_node, run_path, export_value, 0);
        enkf_node_free(enkf_node);
    }
    value_export(export_value);

    value_export_free(export_value);
}

/**
 * @brief Initializes an active run.
 *
 *  * Instantiate res_config_templates which substitutes arg_list from the template
 *      and from run_arg into each template and writes it to runpath;
 *  * substitutes sampled parameters into the parameter nodes and write to runpath;
 *  * substitutes DATAKW into the eclipse data file template and write it to runpath;
 *  * write the job script.
 *
 * @param run_path The runpath string
 * @param iens The realization number.
 * @param fs The file system to write to
 * @param run_id Unique id of run
 * @param subst_list The substitutions to perform for that run.
 */
void init_active_run(const model_config_type *model_config,
                     ensemble_config_type *ens_config,
                     const env_varlist_type *env_varlist, char *run_path,
                     int iens, enkf_fs_type *fs, char *run_id, char *job_name,
                     const subst_list_type *subst_list) {
    ecl_write(ens_config, model_config_get_gen_kw_export_name(model_config),
              run_path, iens, fs);

    // Create the job script
    forward_model_formatted_fprintf(
        model_config_get_forward_model(model_config), run_id, run_path,
        model_config_get_data_root(model_config), subst_list, env_varlist);
}
} // namespace enkf_main

static void enkf_main_copy_ensemble(const ensemble_config_type *ensemble_config,
                                    enkf_fs_type *source_case_fs,
                                    int source_report_step,
                                    enkf_fs_type *target_case_fs,
                                    const std::vector<bool> &iens_mask,
                                    const std::vector<std::string> &node_list) {
    auto &target_state_map = enkf_fs_get_state_map(target_case_fs);

    for (auto &node : node_list) {
        enkf_config_node_type *config_node =
            ensemble_config_get_node(ensemble_config, node.c_str());

        int src_iens = 0;
        for (auto mask : iens_mask) {
            if (mask) {
                node_id_type src_id = {.report_step = source_report_step,
                                       .iens = src_iens};
                node_id_type target_id = {.report_step = 0, .iens = src_iens};

                /* The copy is careful ... */
                if (enkf_config_node_has_node(config_node, source_case_fs,
                                              src_id))
                    enkf_node_copy(config_node, source_case_fs, target_case_fs,
                                   src_id, target_id);

                target_state_map.set(src_iens, STATE_INITIALIZED);
            }
            src_iens++;
        }
    }
}

ERT_CLIB_SUBMODULE("enkf_main", m) {
    using namespace py::literals;
    m.def(
        "init_current_case_from_existing_custom",
        [](Cwrap<ensemble_config_type> ensemble_config,
           Cwrap<enkf_fs_type> source_case, Cwrap<enkf_fs_type> current_case,
           int source_report_step, std::vector<std::string> &node_list,
           std::vector<bool> &iactive) {
            enkf_main_copy_ensemble(ensemble_config, source_case,
                                    source_report_step, current_case, iactive,
                                    node_list);
            enkf_fs_fsync(current_case);
        },
        py::arg("self"), py::arg("source_case"), py::arg("current_case"),
        py::arg("source_report_step"), py::arg("node_list"),
        py::arg("iactive"));
    m.def(
        "init_active_run",
        [](Cwrap<model_config_type> model_config,
           Cwrap<ensemble_config_type> ensemble_config,
           Cwrap<env_varlist_type> env_varlist, char *run_path, int iens,
           Cwrap<enkf_fs_type> sim_fs, char *run_id, char *job_name,
           Cwrap<subst_list_type> subst_list) {
            enkf_main::init_active_run(model_config, ensemble_config,
                                       env_varlist, run_path, iens, sim_fs,
                                       run_id, job_name, subst_list);
        },
        py::arg("model_config"), py::arg("ensemble_config"),
        py::arg("env_varlist"), py::arg("run_path"), py::arg("iens"),
        py::arg("sim_fs"), py::arg("run_id"), py::arg("job_name"),
        py::arg("subst_list"));
    m.def("log_seed", [](Cwrap<rng_type> rng) {
        uint32_t random_seeds[4];
        rng_get_state(rng, reinterpret_cast<char *>(random_seeds));

        std::string random_seed_str;
        for (const auto &random_seed : random_seeds) {
            random_seed_str += fmt::format("{:010}", random_seed);
        }
        logger->info(
            "To repeat this experiment, add the following random seed to "
            "your config file:");
        logger->info("RANDOM_SEED {}", random_seed_str);
    });
}
