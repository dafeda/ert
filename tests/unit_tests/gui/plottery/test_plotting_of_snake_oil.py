from unittest.mock import Mock

import pytest
from qtpy.QtWidgets import QCheckBox

from ert.enkf_main import EnKFMain
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_window import (
    CROSS_ENSEMBLE_STATISTICS,
    DISTRIBUTION,
    ENSEMBLE,
    GAUSSIAN_KDE,
    HISTOGRAM,
    STATISTICS,
    PlotWindow,
)
from ert.services import StorageService
from ert.storage import open_storage


@pytest.fixture
def enkf_main_snake_oil(snake_oil_case_storage):
    yield EnKFMain(snake_oil_case_storage)


@pytest.mark.parametrize(
    "key, plot_name",
    [
        ("FOPR", STATISTICS),
        ("FOPR", ENSEMBLE),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", CROSS_ENSEMBLE_STATISTICS),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", DISTRIBUTION),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", GAUSSIAN_KDE),
        ("SNAKE_OIL_PARAM:OP1_OCTAVES", HISTOGRAM),
    ],
)
def test_that_all_snake_oil_visualisations_matches_snapshot(
    qtbot, enkf_main_snake_oil, plot_name, key
):
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        project=enkf_main_snake_oil.ert_config.ens_path,
    ), open_storage(enkf_main_snake_oil.ert_config.ens_path) as storage:
        gui = _setup_main_window(
            enkf_main_snake_oil, args_mock, GUILogHandler(), storage
        )
        qtbot.addWidget(gui)

        plot_tool = gui.tools["Create plot"]
        plot_tool.trigger()

        qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)
        plot_window = gui.findChild(PlotWindow)
        central_tab = plot_window._central_tab

        # Use an inner function in order for the lifetime
        # of the c++ gui element to not go out before mpl_image_compare
        @pytest.mark.mpl_image_compare(tolerance=10)
        def inner():
            # Cycle through showing all the tabs for all keys
            data_types = plot_window.findChild(DataTypeKeysWidget)
            key_list = data_types.data_type_keys_widget

            found_selected_key = False
            for i in range(key_list.model().rowCount()):
                key_list.setCurrentIndex(key_list.model().index(i, 0))
                selected_key = data_types.getSelectedItem()
                if selected_key.key == key:
                    for i, tab in enumerate(plot_window._plot_widgets):
                        if tab.name == plot_name:
                            found_selected_key = True
                            if central_tab.isTabEnabled(i):
                                central_tab.setCurrentWidget(tab)
                                assert (
                                    selected_key.dimensionality
                                    == tab._plotter.dimensionality
                                )
                                return tab._figure.figure
                            else:
                                assert (
                                    selected_key.dimensionality
                                    != tab._plotter.dimensionality
                                )
            assert found_selected_key

        inner()
        plot_window.close()


def test_that_all_plotter_filter_boxes_yield_expected_filter_results(
    qtbot, enkf_main_snake_oil
):
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        project=enkf_main_snake_oil.ert_config.ens_path,
    ), open_storage(enkf_main_snake_oil.ert_config.ens_path) as storage:
        gui = _setup_main_window(
            enkf_main_snake_oil, args_mock, GUILogHandler(), storage
        )
        gui.notifier.set_storage(storage)
        qtbot.addWidget(gui)

        plot_tool = gui.tools["Create plot"]
        plot_tool.trigger()

        qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)
        plot_window = gui.findChild(PlotWindow)

        key_list = plot_window.findChild(DataTypeKeysWidget).data_type_keys_widget
        item_count = [3, 10, 44]

        assert key_list.model().rowCount() == sum(item_count)
        cbs = plot_window.findChildren(QCheckBox, "FilterCheckBox")

        for i in range(len(item_count)):
            for u, cb in enumerate(cbs):
                cb.setChecked(i == u)

            assert key_list.model().rowCount() in item_count

        plot_window.close()
