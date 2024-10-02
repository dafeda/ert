import pytest
from pytestqt.qt_compat import qt_api
from qtpy.QtCore import QModelIndex
from qtpy.QtGui import QColor

from ert.ensemble_evaluator.state import COLOR_FAILED
from ert.gui.model.snapshot import FMStepColorHint, SnapshotModel

from .gui_models_utils import finish_snapshot


@pytest.mark.integration_test
def test_using_qt_model_tester(qtmodeltester, full_snapshot):
    model = SnapshotModel()

    reporting_mode = qt_api.QtTest.QAbstractItemModelTester.FailureReportingMode.Warning
    tester = qt_api.QtTest.QAbstractItemModelTester(  # noqa: F841, prevent GC
        model, reporting_mode
    )

    model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")
    model._add_snapshot(SnapshotModel.prerender(full_snapshot), "1")

    snapshot = finish_snapshot(SnapshotModel.prerender(full_snapshot))
    model._update_snapshot(SnapshotModel.prerender(snapshot), "0")
    model._update_snapshot(SnapshotModel.prerender(snapshot), "1")

    qtmodeltester.check(model, force_py=True)


def test_realization_sort_order(full_snapshot):
    model = SnapshotModel()

    model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")

    for i in range(0, 100):
        iter_index = model.index(i, 0, model.index(0, 0, QModelIndex()))

        assert str(i) == str(iter_index.internalPointer().id_)
        print(i, iter_index.internalPointer())


def test_realization_state_is_queue_finalized_state(fail_snapshot):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(fail_snapshot), "0")
    first_real = model.index(0, 0, model.index(0, 0))

    color, done_count, full_count = model.data(first_real, FMStepColorHint)
    assert color == QColor(*COLOR_FAILED)
    assert done_count == 1
    assert full_count == 1


def test_snapshot_model_data_intact_on_full_update(full_snapshot, fail_snapshot):
    model = SnapshotModel()
    model._add_snapshot(SnapshotModel.prerender(full_snapshot), "0")

    first_real = model.index(0, 0, model.index(0, 0))
    assert first_real.internalPointer().children["0"].data["status"] == "Pending"
    # Update with a different snapshot, -- data should change accordingly
    model._add_snapshot(SnapshotModel.prerender(fail_snapshot), "0")
    first_real = model.index(0, 0, model.index(0, 0))

    assert first_real.internalPointer().children["0"].data["status"] == "Finished"