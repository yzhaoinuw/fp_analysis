from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.io import loadmat

from fp_analysis_app.event_analysis import Analyses, Event_Utils, Perievent_Plots


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
F268_PATH = DATA_DIR / "F268.mat"
TRANSITIONS_F268_PATH = DATA_DIR / "Transitions_F268.xlsx"
BASELINE_WINDOW = 30
ANALYSIS_WINDOW = 60


class TestPerieventAnalysisWithF268(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mat = loadmat(F268_PATH, squeeze_me=True)
        cls.fp_freq = float(cls.mat["fp_frequency"])
        cls.signal_names = tuple(cls.mat["fp_signal_names"])
        cls.signal_length = len(cls.mat[cls.signal_names[0]])
        cls.duration = int(np.ceil((cls.signal_length - 1) / cls.fp_freq))
        cls.event_utils = Event_Utils(
            cls.fp_freq,
            cls.duration,
            nsec_before=BASELINE_WINDOW,
            nsec_after=ANALYSIS_WINDOW,
        )
        cls.events = cls.event_utils.read_events(event_file=TRANSITIONS_F268_PATH)
        cls.analyses = Analyses(
            fp_freq=cls.fp_freq,
            baseline_window=BASELINE_WINDOW,
        )

    def _get_perievent_signals(self, event_name, signal_name):
        event_time = self.events[event_name]
        perievent_windows = self.event_utils.make_perievent_windows(event_time)
        perievent_indices = self.event_utils.get_perievent_indices(perievent_windows)
        perievent_signals = self.mat[signal_name][perievent_indices]
        return perievent_windows, perievent_indices, perievent_signals

    def test_read_events_filters_empty_columns_and_edge_events(self):
        expected_counts = {
            "sws_wake": 9,
            "wake_sws": 15,
            "REM_wake": 8,
            "sws_MA": 91,
            "sws_REM": 10,
            "REM_MA": 2,
        }

        self.assertEqual(expected_counts, {k: len(v) for k, v in self.events.items()})
        self.assertNotIn("REM_sws", self.events)

        min_allowed_time = BASELINE_WINDOW
        max_allowed_time = self.duration - ANALYSIS_WINDOW
        for event_times in self.events.values():
            self.assertTrue(np.all(event_times >= min_allowed_time))
            self.assertTrue(np.all(event_times <= max_allowed_time))

    def test_perievent_signal_extraction_stays_in_bounds_for_real_fixture(self):
        perievent_windows, perievent_indices, perievent_signals = (
            self._get_perievent_signals("wake_sws", "NE2m")
        )

        self.assertEqual((15, 90), perievent_windows.shape)
        self.assertEqual((15, 91553), perievent_indices.shape)
        self.assertEqual(perievent_indices.shape, perievent_signals.shape)
        self.assertGreaterEqual(int(perievent_indices.min()), 0)
        self.assertLess(int(perievent_indices.max()), self.signal_length)

    def test_auc_analysis_matches_reference_values_for_ne2m_sws_wake(self):
        _, perievent_indices, perievent_signals = self._get_perievent_signals(
            "sws_wake",
            "NE2m",
        )
        result = self.analyses.get_perievent_analyses(perievent_signals)
        event_time_index = round(BASELINE_WINDOW * self.fp_freq)

        np.testing.assert_allclose(
            result["perievent_signals_normalized"][:, event_time_index],
            0.0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            result["reaction_signal_auc"][:5],
            np.array([1.445545, 2.455903, 3.152407, 6.800297, 1.316329]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result["max_peak_magnitude"][:5],
            np.array([3.582563, 5.549205, 5.945642, 8.685765, 4.746928]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result["first_peak_time"][:5],
            np.array([8.0, 8.0, 9.0, 16.0, 9.0]),
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result["decay_time"][:5],
            np.array([39.681392, 59.998862, 44.964249, 59.998862, 39.820984]),
            atol=1e-6,
            equal_nan=True,
        )
        self.assertEqual((9, 91553), perievent_indices.shape)

    def test_mean_trace_workbook_exports_and_appends_subject_columns(self):
        _, _, perievent_signals = self._get_perievent_signals("wake_sws", "NE2m")
        plots = Perievent_Plots(
            self.fp_freq,
            "wake_sws",
            nsec_before=BASELINE_WINDOW,
            nsec_after=ANALYSIS_WINDOW,
        )
        f268_df = plots.build_mean_trace_export_df(
            perievent_signals,
            subject_id="F268",
            downsample_factor=100,
        )
        repeat_df = plots.build_mean_trace_export_df(
            perievent_signals,
            subject_id="F268_repeat",
            downsample_factor=100,
        )

        with TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "NE2m_bw30_aw60.xlsx"
            Perievent_Plots.export_mean_trace_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": f268_df},
            )
            Perievent_Plots.export_mean_trace_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": repeat_df},
            )

            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(["time_s", "F268", "F268_repeat"], exported.columns.tolist())
        self.assertEqual(915, len(exported))
        np.testing.assert_allclose(
            exported["time_s"].head(3),
            np.array([-29.95134, -29.853036, -29.754732]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            exported["F268"].head(3),
            np.array([-0.412663, -0.417773, -0.424488]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            exported["F268"].to_numpy(),
            exported["F268_repeat"].to_numpy(),
            atol=1e-12,
        )

    def test_auc_workbook_aligns_event_index_when_subjects_have_different_counts(self):
        _, _, perievent_signals = self._get_perievent_signals("wake_sws", "NE2m")
        result = self.analyses.get_perievent_analyses(perievent_signals)
        plots = Perievent_Plots(
            self.fp_freq,
            "wake_sws",
            nsec_before=BASELINE_WINDOW,
            nsec_after=ANALYSIS_WINDOW,
        )
        f268_df = plots.build_auc_export_df(
            result["reaction_signal_auc"],
            subject_id="F268",
        )
        short_df = plots.build_auc_export_df(
            result["reaction_signal_auc"][:10],
            subject_id="F268_short",
        )

        with TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "NE2m_auc_bw30_aw60.xlsx"
            Perievent_Plots.export_occurrence_value_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": f268_df},
                index_column="event_index",
            )
            Perievent_Plots.export_occurrence_value_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": short_df},
                index_column="event_index",
            )

            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(["event_index", "F268", "F268_short"], exported.columns.tolist())
        self.assertEqual(list(range(1, 16)), exported["event_index"].tolist())
        np.testing.assert_allclose(
            exported["F268"].head(3),
            np.array([-1.933119, -2.025191, -2.452559]),
            atol=1e-6,
        )
        self.assertTrue(exported["F268_short"].iloc[10:].isna().all())

    def test_max_peak_magnitude_workbook_aligns_event_index_when_subjects_differ(self):
        _, _, perievent_signals = self._get_perievent_signals("wake_sws", "NE2m")
        result = self.analyses.get_perievent_analyses(perievent_signals)
        plots = Perievent_Plots(
            self.fp_freq,
            "wake_sws",
            nsec_before=BASELINE_WINDOW,
            nsec_after=ANALYSIS_WINDOW,
        )
        f268_df = plots.build_occurrence_value_export_df(
            result["max_peak_magnitude"],
            subject_id="F268",
        )
        short_df = plots.build_occurrence_value_export_df(
            result["max_peak_magnitude"][:10],
            subject_id="F268_short",
        )

        with TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "NE2m_max_peak_magnitude_bw30_aw60.xlsx"
            Perievent_Plots.export_occurrence_value_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": f268_df},
                index_column="event_index",
            )
            Perievent_Plots.export_occurrence_value_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": short_df},
                index_column="event_index",
            )

            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(["event_index", "F268", "F268_short"], exported.columns.tolist())
        self.assertEqual(list(range(1, 16)), exported["event_index"].tolist())
        np.testing.assert_allclose(
            exported["F268"].head(3),
            np.array([0.863213, 0.804854, 0.0]),
            atol=1e-6,
        )
        self.assertTrue(exported["F268_short"].iloc[10:].isna().all())

    def test_first_peak_time_workbook_aligns_event_index_when_subjects_differ(self):
        _, _, perievent_signals = self._get_perievent_signals("wake_sws", "NE2m")
        result = self.analyses.get_perievent_analyses(perievent_signals)
        plots = Perievent_Plots(
            self.fp_freq,
            "wake_sws",
            nsec_before=BASELINE_WINDOW,
            nsec_after=ANALYSIS_WINDOW,
        )
        f268_df = plots.build_occurrence_value_export_df(
            result["first_peak_time"],
            subject_id="F268",
        )
        short_df = plots.build_occurrence_value_export_df(
            result["first_peak_time"][:10],
            subject_id="F268_short",
        )

        with TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "NE2m_first_peak_time_bw30_aw60.xlsx"
            Perievent_Plots.export_occurrence_value_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": f268_df},
                index_column="event_index",
            )
            Perievent_Plots.export_occurrence_value_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": short_df},
                index_column="event_index",
            )

            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(["event_index", "F268", "F268_short"], exported.columns.tolist())
        self.assertEqual(list(range(1, 16)), exported["event_index"].tolist())
        np.testing.assert_allclose(
            exported["F268"].head(3),
            np.array([np.nan, np.nan, np.nan]),
            atol=1e-6,
            equal_nan=True,
        )
        self.assertTrue(exported["F268_short"].iloc[10:].isna().all())

    def test_decay_time_workbook_aligns_event_index_when_subjects_differ(self):
        _, _, perievent_signals = self._get_perievent_signals("wake_sws", "NE2m")
        result = self.analyses.get_perievent_analyses(perievent_signals)
        plots = Perievent_Plots(
            self.fp_freq,
            "wake_sws",
            nsec_before=BASELINE_WINDOW,
            nsec_after=ANALYSIS_WINDOW,
        )
        f268_df = plots.build_occurrence_value_export_df(
            result["decay_time"],
            subject_id="F268",
        )
        short_df = plots.build_occurrence_value_export_df(
            result["decay_time"][:10],
            subject_id="F268_short",
        )

        with TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "NE2m_decay_time_bw30_aw60.xlsx"
            Perievent_Plots.export_occurrence_value_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": f268_df},
                index_column="event_index",
            )
            Perievent_Plots.export_occurrence_value_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": short_df},
                index_column="event_index",
            )

            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(["event_index", "F268", "F268_short"], exported.columns.tolist())
        self.assertEqual(list(range(1, 16)), exported["event_index"].tolist())
        np.testing.assert_allclose(
            exported["F268"].head(3),
            np.array([np.nan, np.nan, np.nan]),
            atol=1e-6,
            equal_nan=True,
        )
        self.assertTrue(exported["F268"].iloc[:6].isna().all())
        self.assertAlmostEqual(exported["F268"].iloc[6], 59.998862, places=6)
        self.assertTrue(exported["F268_short"].iloc[10:].isna().all())

    def test_cross_correlation_export_df_uses_mean_trace_per_lag(self):
        lags_time = np.array([-1.0, 0.0, 1.0])
        mean_corr = np.array([0.2, 0.5, 0.8])

        exported = Perievent_Plots.build_cross_correlation_export_df(
            lags_time=lags_time,
            mean_corr=mean_corr,
            subject_id="F268",
        )

        self.assertEqual(["lag_s", "F268"], exported.columns.tolist())
        np.testing.assert_allclose(
            exported["lag_s"].to_numpy(),
            np.array([-1.0, 0.0, 1.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268"].to_numpy(),
            np.array([0.2, 0.5, 0.8]),
            atol=1e-12,
        )

    def test_summarize_cross_correlation_downsamples_only_derived_traces(self):
        plots = Perievent_Plots(
            self.fp_freq,
            "wake_sws",
            nsec_before=BASELINE_WINDOW,
            nsec_after=ANALYSIS_WINDOW,
        )
        lags_time = np.array([-1.5, -0.5, 0.5, 1.5])
        cross_correlations = np.array(
            [
                [0.0, 0.2, 0.4, 0.6],
                [0.2, 0.4, 0.6, 0.8],
            ]
        )

        lags_downsampled, mean_corr, se_corr = plots.summarize_cross_correlation(
            lags_time,
            cross_correlations,
            downsample_factor=2,
        )

        np.testing.assert_allclose(
            lags_downsampled,
            np.array([-1.0, 1.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            mean_corr,
            np.array([0.2, 0.6]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            se_corr,
            np.array([0.070710678, 0.070710678]),
            atol=1e-9,
        )

    def test_cross_correlation_workbook_exports_and_appends_subject_columns(self):
        f268_df = Perievent_Plots.build_cross_correlation_export_df(
            lags_time=np.array([-1.0, 0.0, 1.0]),
            mean_corr=np.array([0.2, 0.5, 0.8]),
            subject_id="F268",
        )
        repeat_df = Perievent_Plots.build_cross_correlation_export_df(
            lags_time=np.array([-1.0, 0.0, 1.0]),
            mean_corr=np.array([0.1, 0.3, 0.5]),
            subject_id="F268_repeat",
        )

        with TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "NE2m_mClY_cross_correlation_bw30_aw60.xlsx"
            Perievent_Plots.export_cross_correlation_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": f268_df},
            )
            Perievent_Plots.export_cross_correlation_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": repeat_df},
            )

            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(
            ["lag_s", "F268", "F268_repeat"],
            exported.columns.tolist(),
        )
        np.testing.assert_allclose(
            exported["lag_s"].to_numpy(),
            np.array([-1.0, 0.0, 1.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268"].to_numpy(),
            np.array([0.2, 0.5, 0.8]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268_repeat"].to_numpy(),
            np.array([0.1, 0.3, 0.5]),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
