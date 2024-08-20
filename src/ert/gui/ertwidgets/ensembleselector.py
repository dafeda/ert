from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox

from ert.gui.ertnotifier import ErtNotifier
from ert.storage.realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    from ert.storage import Ensemble


class EnsembleSelector(QComboBox):
    ensemble_populated = Signal()

    def __init__(
        self,
        notifier: ErtNotifier,
        update_ert: bool = True,
        show_only_undefined: bool = False,
        show_only_no_children: bool = False,
    ):
        super().__init__()
        self.notifier = notifier
        self._update_ert = update_ert
        self._show_only_undefined = show_only_undefined
        # If True, we filter out any ensembles which have children
        # One use case is if a user wants to rerun because of failures
        # not related to parameterization. We can allow that, but only
        # if the ensemble has not been used in an update, as that would
        # invalidate the result.
        self._show_only_no_children = show_only_no_children
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.setEnabled(False)

        if update_ert:
            self.currentIndexChanged[int].connect(self._on_current_index_changed)
            notifier.current_ensemble_changed.connect(
                self._on_global_current_ensemble_changed
            )

        notifier.ertChanged.connect(self.populate)
        notifier.storage_changed.connect(self.populate)

        if notifier.is_storage_available:
            self.populate()

    @property
    def selected_ensemble(self) -> Ensemble:
        return self.itemData(self.currentIndex())

    def populate(self) -> None:
        block = self.blockSignals(True)
        self.clear()
        if self._ensemble_list():
            self.setEnabled(True)
        for ensemble in self._ensemble_list():
            self.addItem(ensemble.name, userData=ensemble)
        current_index = self.findData(
            self.notifier.current_ensemble, Qt.ItemDataRole.UserRole
        )
        self.setCurrentIndex(max(current_index, 0))
        self.blockSignals(block)
        self.ensemble_populated.emit()

    def _ensemble_list(self) -> Iterable[Ensemble]:
        if not self.notifier.is_storage_available:
            return []

        all_ensembles = []
        for experiment in self.notifier.storage.experiments:
            all_ensembles.extend(experiment.ensembles)

        if self._show_only_undefined:
            all_ensembles = [
                ensemble
                for ensemble in all_ensembles
                if all(
                    e == RealizationStorageState.UNDEFINED
                    for e in ensemble.get_ensemble_state()
                )
            ]

        if self._show_only_no_children:
            parents = [ens.parent for ens in all_ensembles if ens.parent]
            all_ensembles = [val for val in all_ensembles if val.id not in parents]

        return sorted(all_ensembles, key=lambda x: x.started_at, reverse=True)

    def _on_current_index_changed(self, index: int) -> None:
        self.notifier.set_current_ensemble(self.itemData(index))

    def _on_global_current_ensemble_changed(self, data: Optional[Ensemble]) -> None:
        self.setCurrentIndex(max(self.findData(data, Qt.ItemDataRole.UserRole), 0))
