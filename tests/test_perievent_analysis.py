from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.io import loadmat

from fp_analysis_app.event_analysis import Analyses, Event_Utils, Perievent_Plots
from fp_analysis_app.export_settings import (
    build_analysis_config_dirname,
    build_analysis_description_text,
    get_analysis_export_dir,
    write_analysis_description_file,
)
from fp_analysis_app.mat_utils import (
    get_fp_signal_names,
    get_visualization_signal_data,
    get_visualization_signal_names_and_frequency,
)
from fp_analysis_app.sleep_event_import import is_sleep_bout_table


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
F268_PATH = DATA_DIR / "F268.mat"
TRANSITIONS_F268_PATH = DATA_DIR / "Transitions_F268.xlsx"
BASELINE_WINDOW = 30
ANALYSIS_WINDOW = 60


class TestSleepBoutTableImport(unittest.TestCase):
    def setUp(self):
        self.event_utils = Event_Utils(
            fp_freq=1,
            duration=400,
            nsec_before=30,
            nsec_after=60,
        )

    def test_detects_sleep_bout_table_format(self):
        df = pd.DataFrame(
            {
                "Unnamed: 0": [0, 1],
                "sleep_scores": [1, 2],
                "start": [0, 50],
                "end": [49, 99],
                "duration": [50, 50],
            }
        )

        self.assertTrue(is_sleep_bout_table(df))

    def test_converts_one_based_sleep_scores_to_transition_events(self):
        df = pd.DataFrame(
            {
                "index": [0, 1, 2, 3, 4, 5],
                "sleep_scores": [1, 2, 3, 4, 2, 1],
                "start": [0, 40, 80, 120, 160, 220],
                "end": [39, 79, 119, 159, 219, 259],
                "duration": [40, 40, 40, 40, 60, 40],
            }
        )

        events = self.event_utils.read_events(df_events=df)

        self.assertEqual(
            {
                "wake_nrem": [40],
                "nrem_rem": [80],
                "rem_ma": [120],
                "ma_nrem": [160],
                "nrem_wake": [220],
            },
            {key: value.tolist() for key, value in events.items()},
        )

    def test_converts_zero_based_sleep_scores_to_transition_events(self):
        df = pd.DataFrame(
            {
                "sleep_scores": [0, 1, 2, 3],
                "start": [35, 70, 110, 150],
                "end": [69, 109, 149, 189],
                "duration": [35, 40, 40, 40],
            }
        )

        events = self.event_utils.read_events(df_events=df)

        self.assertEqual(
            {
                "wake_nrem": [70],
                "nrem_rem": [110],
                "rem_ma": [150],
            },
            {key: value.tolist() for key, value in events.items()},
        )

    def test_filters_transition_times_using_existing_event_window_rules(self):
        df = pd.DataFrame(
            {
                "sleep_scores": [1, 2, 3, 1],
                "start": [0, 20, 100, 360],
                "end": [19, 99, 359, 399],
                "duration": [20, 80, 260, 40],
            }
        )

        events = self.event_utils.read_events(df_events=df)

        self.assertEqual(
            {"nrem_rem": [100]},
            {key: value.tolist() for key, value in events.items()},
        )


class TestMakeFigureFallbacks(unittest.TestCase):
    def test_visualization_signal_helper_uses_ne_when_fp_signal_names_are_missing(self):
        mat = {
            "ne": np.array([0.0, 0.5, -0.25, 0.75]),
            "ne_frequency": np.array(2.0),
            "start_time": 0,
        }

        signal_names, frequency = get_visualization_signal_names_and_frequency(mat)

        self.assertEqual(["ne"], signal_names)
        self.assertEqual(2.0, frequency)

    def test_visualization_signal_data_helper_returns_ne_signal(self):
        mat = {
            "ne": np.array([0.0, 0.5, -0.25, 0.75]),
            "ne_frequency": np.array(2.0),
        }

        signal_names, signals, frequency = get_visualization_signal_data(mat)

        self.assertEqual(["ne"], signal_names)
        np.testing.assert_array_equal(np.array([0.0, 0.5, -0.25, 0.75]), signals[0])
        self.assertEqual(2.0, frequency)


@unittest.skipUnless(
    F268_PATH.exists() and TRANSITIONS_F268_PATH.exists(),
    "Local F268 fixture files are required for these integration tests.",
)
class TestPerieventAnalysisWithF268(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mat = loadmat(F268_PATH, squeeze_me=True)
        cls.fp_freq = float(cls.mat["fp_frequency"])
        cls.signal_names = tuple(get_fp_signal_names(cls.mat))
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

        self.assertEqual(
            [
                "time_s",
                "F268_mean",
                "F268_sd",
                "F268_n",
                "F268_repeat_mean",
                "F268_repeat_sd",
                "F268_repeat_n",
            ],
            exported.columns.tolist(),
        )
        self.assertEqual(915, len(exported))
        np.testing.assert_allclose(
            exported["time_s"].head(3),
            np.array([-29.95134, -29.853036, -29.754732]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            exported["F268_mean"].head(3),
            np.array([-0.412663, -0.417773, -0.424488]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            exported["F268_sd"].head(3),
            np.array([3.088589, 3.086347, 3.082354]),
            atol=1e-6,
        )
        self.assertTrue((exported["F268_n"] == 15).all())
        np.testing.assert_allclose(
            exported["F268_mean"].to_numpy(),
            exported["F268_repeat_mean"].to_numpy(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268_sd"].to_numpy(),
            exported["F268_repeat_sd"].to_numpy(),
            atol=1e-12,
        )
        self.assertTrue((exported["F268_repeat_n"] == 15).all())

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

class TestPerieventPlotExports(unittest.TestCase):
    FP_FREQ = 1017.25

    def test_cross_correlation_export_df_uses_mean_trace_per_lag(self):
        lags_time = np.array([-1.0, 0.0, 1.0])
        mean_corr = np.array([0.2, 0.5, 0.8])

        exported = Perievent_Plots.build_cross_correlation_export_df(
            lags_time=lags_time,
            mean_corr=mean_corr,
            std_corr=np.array([0.1, 0.2, 0.3]),
            n_occurrences=4,
            subject_id="F268",
        )

        self.assertEqual(
            ["lag_s", "F268_mean", "F268_sd", "F268_n"],
            exported.columns.tolist(),
        )
        np.testing.assert_allclose(
            exported["lag_s"].to_numpy(),
            np.array([-1.0, 0.0, 1.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268_mean"].to_numpy(),
            np.array([0.2, 0.5, 0.8]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268_sd"].to_numpy(),
            np.array([0.1, 0.2, 0.3]),
            atol=1e-12,
        )
        self.assertTrue((exported["F268_n"] == 4).all())

    def test_lag_at_strongest_cross_correlation_uses_largest_magnitude(self):
        strongest_lag_s = Perievent_Plots.get_lag_at_strongest_cross_correlation(
            lags_time=np.array([-1.0, 0.0, 1.0]),
            cross_correlations=np.array(
                [
                    [0.2, -0.7, 0.4],
                    [0.1, 0.3, 0.25],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )

        np.testing.assert_allclose(
            strongest_lag_s[:2],
            np.array([0.0, 0.0]),
            atol=1e-12,
        )
        self.assertTrue(np.isnan(strongest_lag_s[2]))

    def test_strongest_cross_correlation_export_df_uses_event_index_per_occurrence(self):
        strongest_lag_s = np.array([-1.5, 0.0, 1.5])

        exported = Perievent_Plots.build_strongest_cross_correlation_export_df(
            strongest_lag_s=strongest_lag_s,
            subject_id="F268",
        )

        self.assertEqual(["event_index", "F268"], exported.columns.tolist())
        self.assertEqual([1, 2, 3], exported["event_index"].tolist())
        np.testing.assert_allclose(
            exported["F268"].to_numpy(),
            np.array([-1.5, 0.0, 1.5]),
            atol=1e-12,
        )

    def test_summarize_cross_correlation_downsamples_only_derived_traces(self):
        plots = Perievent_Plots(
            self.FP_FREQ,
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
            std_corr=np.array([0.05, 0.1, 0.15]),
            n_occurrences=3,
            subject_id="F268",
        )
        repeat_df = Perievent_Plots.build_cross_correlation_export_df(
            lags_time=np.array([-1.0, 0.0, 1.0]),
            mean_corr=np.array([0.1, 0.3, 0.5]),
            std_corr=np.array([0.02, 0.04, 0.06]),
            n_occurrences=2,
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
            [
                "lag_s",
                "F268_mean",
                "F268_sd",
                "F268_n",
                "F268_repeat_mean",
                "F268_repeat_sd",
                "F268_repeat_n",
            ],
            exported.columns.tolist(),
        )
        np.testing.assert_allclose(
            exported["lag_s"].to_numpy(),
            np.array([-1.0, 0.0, 1.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268_mean"].to_numpy(),
            np.array([0.2, 0.5, 0.8]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268_sd"].to_numpy(),
            np.array([0.05, 0.1, 0.15]),
            atol=1e-12,
        )
        self.assertTrue((exported["F268_n"] == 3).all())
        np.testing.assert_allclose(
            exported["F268_repeat_mean"].to_numpy(),
            np.array([0.1, 0.3, 0.5]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268_repeat_sd"].to_numpy(),
            np.array([0.02, 0.04, 0.06]),
            atol=1e-12,
        )
        self.assertTrue((exported["F268_repeat_n"] == 2).all())

    def test_strongest_cross_correlation_workbook_aligns_event_index_when_subjects_differ(self):
        f268_df = Perievent_Plots.build_strongest_cross_correlation_export_df(
            strongest_lag_s=np.array([-1.5, 0.0, 1.5]),
            subject_id="F268",
        )
        short_df = Perievent_Plots.build_strongest_cross_correlation_export_df(
            strongest_lag_s=np.array([-0.5, 0.5]),
            subject_id="F268_short",
        )

        with TemporaryDirectory() as tmpdir:
            workbook_path = (
                Path(tmpdir)
                / "NE2m_mClY_strongest_cross_correlation_time_lag_bw30_aw60.xlsx"
            )
            Perievent_Plots.export_strongest_cross_correlation_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": f268_df},
            )
            Perievent_Plots.export_strongest_cross_correlation_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": short_df},
            )

            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(
            ["event_index", "F268", "F268_short"],
            exported.columns.tolist(),
        )
        self.assertEqual([1, 2, 3], exported["event_index"].tolist())
        np.testing.assert_allclose(
            exported["F268"].to_numpy(),
            np.array([-1.5, 0.0, 1.5]),
            atol=1e-12,
        )
        self.assertTrue(exported["F268_short"].iloc[2:].isna().all())

    def test_cross_correlation_workbook_overwrites_existing_subject_columns(self):
        first_df = Perievent_Plots.build_cross_correlation_export_df(
            lags_time=np.array([-1.0, 0.0, 1.0]),
            mean_corr=np.array([0.2, 0.5, 0.8]),
            std_corr=np.array([0.05, 0.1, 0.15]),
            n_occurrences=3,
            subject_id="F268",
        )
        replacement_df = Perievent_Plots.build_cross_correlation_export_df(
            lags_time=np.array([-1.0, 0.0, 1.0]),
            mean_corr=np.array([0.3, 0.6, 0.9]),
            std_corr=np.array([0.06, 0.11, 0.16]),
            n_occurrences=4,
            subject_id="F268",
        )

        with TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "NE2m_mClY_cross_correlation_bw30_aw60.xlsx"
            Perievent_Plots.export_cross_correlation_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": first_df},
            )
            Perievent_Plots.export_cross_correlation_workbook(
                workbook_save_path=workbook_path,
                event_sheet_dfs={"wake_sws": replacement_df},
            )

            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(
            ["lag_s", "F268_mean", "F268_sd", "F268_n"],
            exported.columns.tolist(),
        )
        np.testing.assert_allclose(
            exported["F268_mean"].to_numpy(),
            np.array([0.3, 0.6, 0.9]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            exported["F268_sd"].to_numpy(),
            np.array([0.06, 0.11, 0.16]),
            atol=1e-12,
        )
        self.assertTrue((exported["F268_n"] == 4).all())


class TestAnalysisExportSettings(unittest.TestCase):
    def test_build_analysis_config_dirname_sorts_signal_names(self):
        dirname = build_analysis_config_dirname(
            selected_signals=("mClY", "NE2m"),
            baseline_window=30,
            analysis_window=60,
        )

        self.assertEqual("NE2m_mClY_bw30_aw60", dirname)

    def test_get_analysis_export_dir_uses_sorted_signal_folder_name(self):
        export_dir = get_analysis_export_dir(
            base_dir=Path("C:/tmp/exports"),
            selected_signals=("mClY", "NE2m"),
            baseline_window=30,
            analysis_window=60,
        )

        self.assertEqual(
            Path("C:/tmp/exports/NE2m_mClY_bw30_aw60"),
            export_dir,
        )

    def test_build_analysis_description_text_lists_sorted_signals_and_events(self):
        description_text = build_analysis_description_text(
            mat_filepaths=[
                Path("C:/data/F268.mat"),
                Path("C:/data/F269.mat"),
            ],
            export_dir=Path("C:/data/NE2m_mClY_bw30_aw60"),
            selected_signals=("mClY", "NE2m"),
            baseline_window=30,
            analysis_window=60,
            event_names=["wake_sws", "sws_wake"],
        )

        self.assertIn(
            "Selected signals (sorted folder key): NE2m, mClY",
            description_text,
        )
        self.assertIn("Baseline window (s): 30", description_text)
        self.assertIn("Analysis window (s): 60", description_text)
        self.assertIn("Event types: wake_sws, sws_wake", description_text)
        self.assertIn("Source MAT paths:", description_text)
        normalized_description_text = description_text.replace("\\", "/")
        self.assertIn("- C:/data/F268.mat", normalized_description_text)
        self.assertIn("- C:/data/F269.mat", normalized_description_text)

    def test_write_analysis_description_file_appends_new_mat_paths(self):
        with TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "NE2m_mClY_bw30_aw60"
            write_analysis_description_file(
                export_dir=export_dir,
                mat_filepath=Path("C:/data/F268.mat"),
                selected_signals=("mClY", "NE2m"),
                baseline_window=30,
                analysis_window=60,
                event_names=["wake_sws"],
            )
            write_analysis_description_file(
                export_dir=export_dir,
                mat_filepath=Path("C:/data/F269.mat"),
                selected_signals=("mClY", "NE2m"),
                baseline_window=30,
                analysis_window=60,
                event_names=["wake_sws", "sws_wake"],
            )

            description_text = (export_dir / "data_description.txt").read_text(
                encoding="utf-8"
            )

        normalized_description_text = description_text.replace("\\", "/")
        self.assertEqual(1, normalized_description_text.count("- C:/data/F268.mat"))
        self.assertEqual(1, normalized_description_text.count("- C:/data/F269.mat"))
        self.assertIn("Event types: wake_sws, sws_wake", description_text)

if __name__ == "__main__":
    unittest.main()
