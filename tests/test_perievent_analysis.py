from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from fp_analysis_app.event_analysis import (
    Analyses,
    Event_Utils,
    Perievent_Plots,
    get_adaptive_time_axis_settings,
)
from fp_analysis_app.event_editing import (
    EVENT_TIME_NAMES_MAT_FIELD,
    EVENT_TIME_VALUES_MAT_FIELD,
    event_time_dict_to_store_data,
    event_time_dict_from_mat,
    event_time_dict_from_mat_arrays,
    event_time_dict_to_mat_arrays,
    is_remove_event_key,
    mat_has_saved_event_time_dict,
    remove_event_times_in_span,
    selected_data_to_x_span,
    store_data_to_event_time_dict,
    write_event_time_dict_to_mat,
)
from fp_analysis_app.make_figure import EVENT_TIMESTAMP_TRACE_PREFIX, make_figure
from fp_analysis_app.analysis_export import (
    get_analysis_type_checklist_values,
    write_analysis_workbooks,
)
from fp_analysis_app.components_dev import (
    Components,
    get_analysis_signal_select_style,
    get_max_analysis_window,
    is_valid_analysis_signal_selection,
    validate_and_normalize_analysis_windows,
)
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
    has_embedded_event_data,
)
from fp_analysis_app.sleep_event_import import is_sleep_bout_table


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
F268_PATH = DATA_DIR / "F268.mat"
TRANSITIONS_F268_PATH = DATA_DIR / "Transitions_F268.xlsx"
BASELINE_WINDOW = 30
ANALYSIS_WINDOW = 60


def find_component_by_id(component, component_id):
    if isinstance(component, (list, tuple)):
        for child in component:
            result = find_component_by_id(child, component_id)
            if result is not None:
                return result
        return None

    if getattr(component, "id", None) == component_id:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None
    return find_component_by_id(children, component_id)


class TestAnalysisPageSignalHighlight(unittest.TestCase):
    def test_analysis_page_highlights_signal_dropdown_and_disables_results(self):
        children = Components().fill_analysis_page(
            event_names=["wake_nrem"],
            event_count_records=[{"event": "wake_nrem", "count": 3}],
            signal_names=["NE2m", "mClY"],
        )

        removed_text_callout = find_component_by_id(children, "analysis-signal-prompt")
        controls_row = find_component_by_id(children, "analysis-controls-row")
        wrapper = find_component_by_id(children, "signal-select-wrapper")
        show_results_button = find_component_by_id(children, "show-results-button")

        self.assertIsNone(removed_text_callout)
        self.assertIsNotNone(controls_row)
        self.assertEqual("center", controls_row.style["alignItems"])
        self.assertIsNotNone(wrapper)
        self.assertEqual("2px solid #c62828", wrapper.style["border"])
        self.assertIsNotNone(show_results_button)
        self.assertTrue(show_results_button.disabled)

    def test_signal_selection_validation_accepts_only_one_or_two_signals(self):
        self.assertFalse(is_valid_analysis_signal_selection(None))
        self.assertFalse(is_valid_analysis_signal_selection([]))
        self.assertTrue(is_valid_analysis_signal_selection(["NE2m"]))
        self.assertTrue(is_valid_analysis_signal_selection(["NE2m", "mClY"]))
        self.assertFalse(
            is_valid_analysis_signal_selection(["NE2m", "mClY", "GCaMP"])
        )
        self.assertEqual(
            "2px solid transparent",
            get_analysis_signal_select_style(["NE2m"])["border"],
        )
        self.assertEqual(
            "2px solid #c62828",
            get_analysis_signal_select_style(["NE2m", "mClY", "GCaMP"])["border"],
        )

    def test_analysis_page_can_render_after_all_events_are_removed(self):
        children = Components().fill_analysis_page(
            event_names=[],
            event_count_records=[],
            signal_names=["NE2m"],
        )

        event_tabs = find_component_by_id(children, "event-tabs")

        self.assertIsNotNone(event_tabs)
        self.assertEqual("none", event_tabs.value)

    def test_analysis_windows_are_positive_integer_inputs_capped_by_duration(self):
        children = Components().fill_analysis_page(
            event_names=["wake_nrem"],
            event_count_records=[{"event": "wake_nrem", "count": 3}],
            signal_names=["NE2m"],
            recording_duration=400,
        )

        baseline_input = find_component_by_id(children, "baseline-window-input")
        analysis_input = find_component_by_id(children, "analysis-window-input")
        duration_store = find_component_by_id(
            children, "analysis-recording-duration"
        )

        self.assertIsNone(
            find_component_by_id(children, "baseline-window-dropdown")
        )
        self.assertIsNone(
            find_component_by_id(children, "analysis-window-dropdown")
        )
        for window_input in (baseline_input, analysis_input):
            self.assertEqual("number", window_input.type)
            self.assertEqual(1, window_input.min)
            self.assertEqual(1, window_input.step)
            self.assertEqual(99, window_input.max)
        self.assertEqual(30, baseline_input.value)
        self.assertEqual(60, analysis_input.value)
        self.assertEqual(400, duration_store.data)

    def test_analysis_window_default_is_capped_below_one_quarter_duration(self):
        children = Components().fill_analysis_page(
            event_names=["wake_nrem"],
            event_count_records=[{"event": "wake_nrem", "count": 3}],
            signal_names=["NE2m"],
            recording_duration=240,
        )

        analysis_input = find_component_by_id(children, "analysis-window-input")

        self.assertEqual(59, analysis_input.max)
        self.assertEqual(59, analysis_input.value)

    def test_analysis_window_validation_requires_positive_integers(self):
        invalid_values = [None, "", 0, -1, 1.5, float("nan"), True]

        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                baseline, analysis, error = (
                    validate_and_normalize_analysis_windows(
                        invalid_value,
                        60,
                        recording_duration=400,
                    )
                )
                self.assertIsNone(baseline)
                self.assertEqual(60, analysis)
                self.assertIn("positive whole numbers", error)

    def test_analysis_window_validation_rejects_quarter_duration_or_larger(self):
        self.assertEqual(99, get_max_analysis_window(400))

        baseline, analysis, error = validate_and_normalize_analysis_windows(
            100,
            60,
            recording_duration=400,
        )

        self.assertEqual(100, baseline)
        self.assertEqual(60, analysis)
        self.assertIn("maximum 99 seconds", error)

        self.assertEqual(
            (99, 60, ""),
            validate_and_normalize_analysis_windows(
                99,
                60,
                recording_duration=400,
            ),
        )


class TestAdaptivePerieventTimeTicks(unittest.TestCase):
    def test_default_window_keeps_ten_second_tick_spacing(self):
        settings = get_adaptive_time_axis_settings(30, 60)

        np.testing.assert_allclose(np.diff(settings["ticks"]), 10)
        self.assertEqual(10, settings["font_size"])
        self.assertEqual(0, settings["rotation"])

    def test_120_second_analysis_window_reduces_tick_density(self):
        settings = get_adaptive_time_axis_settings(30, 120)
        old_fixed_tick_count = len(np.arange(-30, 121, 10))

        self.assertLess(len(settings["ticks"]), old_fixed_tick_count)
        self.assertLessEqual(len(settings["ticks"]), 9)
        self.assertTrue(np.any(np.isclose(settings["ticks"], 0)))
        self.assertEqual(9, settings["font_size"])
        self.assertEqual(0, settings["rotation"])

    def test_large_window_uses_fewer_tilted_labels(self):
        settings = get_adaptive_time_axis_settings(300, 900)

        self.assertLessEqual(len(settings["ticks"]), 7)
        self.assertEqual(8, settings["font_size"])
        self.assertEqual(45, settings["rotation"])

    def test_axis_formatter_applies_adaptive_ticks(self):
        plots = Perievent_Plots(
            fp_freq=10,
            event="wake_nrem",
            nsec_before=30,
            nsec_after=120,
        )
        expected_ticks = get_adaptive_time_axis_settings(30, 120)["ticks"]
        fig, ax = plt.subplots()
        try:
            plots._format_time_axis(ax)

            np.testing.assert_allclose(ax.get_xticks(), expected_ticks)
            self.assertEqual((-30, 120), tuple(ax.get_xlim()))
        finally:
            plt.close(fig)


class TestFullRecordingEventTimestampLines(unittest.TestCase):
    def _make_synthetic_mat(self):
        time = np.linspace(0, 2 * np.pi, 100)
        return {
            "fp_signal_names": ["NE2m", "mClY"],
            "fp_frequency": 1,
            "NE2m": np.sin(time),
            "mClY": np.cos(time),
        }

    def _event_timestamp_traces(self, fig):
        return [
            trace
            for trace in fig.data
            if str(trace.name).startswith(EVENT_TIMESTAMP_TRACE_PREFIX)
        ]

    def _event_legend_traces(self, fig):
        return [
            trace
            for trace in fig.data
            if trace.showlegend
            and not str(trace.name).startswith(EVENT_TIMESTAMP_TRACE_PREFIX)
        ]

    def _event_legend_annotations(self, fig):
        return [
            annotation
            for annotation in fig.layout.annotations
            if annotation.text
            and "&#9632;" in annotation.text
        ]

    def test_full_recording_figure_shows_all_imported_event_times(self):
        fig = make_figure(
            self._make_synthetic_mat(),
            event_time_dict={
                "wake_nrem": np.array([20, 45]),
                "nrem_rem": np.array([60]),
            },
        )

        event_traces = self._event_timestamp_traces(fig)

        self.assertEqual(4, len(event_traces))
        self.assertEqual(
            {
                f"{EVENT_TIMESTAMP_TRACE_PREFIX}wake_nrem",
                f"{EVENT_TIMESTAMP_TRACE_PREFIX}nrem_rem",
            },
            {trace.name for trace in event_traces},
        )
        self.assertTrue(
            any(
                np.allclose(
                    trace.x,
                    [20.0, 20.0, np.nan, 45.0, 45.0, np.nan],
                    equal_nan=True,
                )
                for trace in event_traces
                if trace.name == f"{EVENT_TIMESTAMP_TRACE_PREFIX}wake_nrem"
            )
        )
        legend_traces = self._event_legend_traces(fig)
        legend_annotations = self._event_legend_annotations(fig)
        self.assertEqual(
            [],
            legend_traces,
        )
        self.assertEqual(2, len(legend_annotations))
        self.assertEqual(
            {"wake_nrem", "nrem_rem"},
            {
                next(
                    event_name
                    for event_name in ("wake_nrem", "nrem_rem")
                    if event_name in annotation.text
                )
                for annotation in legend_annotations
            },
        )
        self.assertTrue(
            all("<span" in annotation.text for annotation in legend_annotations)
        )
        self.assertTrue(all(trace.showlegend is False for trace in event_traces))

    def test_event_legend_keeps_three_foot_shock_labels_on_one_row(self):
        fig = make_figure(
            self._make_synthetic_mat(),
            event_time_dict={
                "foot_shock_0.1s": np.array([20]),
                "foot_shock_0.5s": np.array([45]),
                "foot_shock_1.0s": np.array([60]),
            },
        )

        legend_annotations = self._event_legend_annotations(fig)

        self.assertEqual(3, len(legend_annotations))
        self.assertEqual(
            {"foot_shock_0.1s", "foot_shock_0.5s", "foot_shock_1.0s"},
            {
                next(
                    event_name
                    for event_name in (
                        "foot_shock_0.1s",
                        "foot_shock_0.5s",
                        "foot_shock_1.0s",
                    )
                    if event_name in annotation.text
                )
                for annotation in legend_annotations
            },
        )
        self.assertEqual({legend_annotations[0].y}, {a.y for a in legend_annotations})
        self.assertLess(
            max(a.x for a in legend_annotations) - min(a.x for a in legend_annotations),
            0.17,
        )

    def test_full_recording_figure_omits_removed_event_times(self):
        fig = make_figure(
            self._make_synthetic_mat(),
            event_time_dict={"wake_nrem": np.array([45])},
        )

        event_traces = self._event_timestamp_traces(fig)

        self.assertEqual(2, len(event_traces))
        self.assertTrue(
            all(
                np.allclose(trace.x, [45.0, 45.0, np.nan], equal_nan=True)
                for trace in event_traces
            )
        )

    def test_can_toggle_expanded_event_window_coloring(self):
        labels = np.full(99, np.nan)
        labels[10:20] = 0
        label_dict = {"label_names": ["wake_nrem"], "labels": labels}

        hidden_fig = make_figure(
            self._make_synthetic_mat(),
            label_dict=label_dict,
            event_time_dict={"wake_nrem": np.array([15])},
            show_period_labels=False,
        )
        shown_fig = make_figure(
            self._make_synthetic_mat(),
            label_dict=label_dict,
            event_time_dict={"wake_nrem": np.array([15])},
            show_period_labels=True,
        )

        hidden_period_traces = [
            trace for trace in hidden_fig.data if trace.name == "Period Labels"
        ]
        shown_period_traces = [
            trace for trace in shown_fig.data if trace.name == "Period Labels"
        ]

        self.assertTrue(
            all(
                np.isnan(np.asarray(trace.z, dtype=float)).all()
                for trace in hidden_period_traces
            )
        )
        self.assertTrue(
            any(
                not np.isnan(np.asarray(trace.z, dtype=float)).all()
                for trace in shown_period_traces
            )
        )


class TestAnnotationModeEventDeletion(unittest.TestCase):
    def test_extracts_x_span_from_rectangle_selection(self):
        selected_data = {"range": {"x": [42.5, 20.0]}}

        self.assertEqual((20.0, 42.5), selected_data_to_x_span(selected_data))

    def test_extracts_x_span_from_selected_points_when_range_is_missing(self):
        selected_data = {"points": [{"x": 30}, {"x": 20}, {"x": 25}]}

        self.assertEqual((20.0, 30.0), selected_data_to_x_span(selected_data))

    def test_delete_and_backspace_remove_event_timestamps(self):
        self.assertTrue(is_remove_event_key({"key": "Delete"}))
        self.assertTrue(is_remove_event_key({"key": "Backspace"}))
        self.assertFalse(is_remove_event_key({"key": "Enter"}))
        self.assertFalse(is_remove_event_key(None))

    def test_filters_event_times_inside_selected_span(self):
        event_time_dict = {
            "wake_nrem": np.array([10, 20, 30, 40]),
            "nrem_rem": np.array([15, 35]),
        }

        filtered_events, removed_count = remove_event_times_in_span(
            event_time_dict,
            (20, 35),
        )

        self.assertEqual(3, removed_count)
        self.assertEqual(
            {"wake_nrem": [10, 40], "nrem_rem": [15]},
            {event: times.tolist() for event, times in filtered_events.items()},
        )

    def test_filter_drops_event_names_when_all_times_are_removed(self):
        filtered_events, removed_count = remove_event_times_in_span(
            {"wake_nrem": np.array([10, 12])},
            (0, 20),
        )

        self.assertEqual(2, removed_count)
        self.assertEqual({}, filtered_events)

    def test_event_time_store_round_trip_uses_json_friendly_lists(self):
        event_time_dict = {
            "wake_nrem": np.array([10, 20]),
            "nrem_rem": np.array([30]),
        }

        store_data = event_time_dict_to_store_data(event_time_dict)
        restored = store_data_to_event_time_dict(store_data)

        self.assertEqual({"wake_nrem": [10, 20], "nrem_rem": [30]}, store_data)
        self.assertEqual(
            {"wake_nrem": [10, 20], "nrem_rem": [30]},
            {event: times.tolist() for event, times in restored.items()},
        )

    def test_event_time_mat_arrays_round_trip_preserves_event_labels(self):
        event_time_dict = {
            "wake_nrem": np.array([10, 20]),
            "foot_shock_0.1s": np.array([30.5]),
        }

        event_names, event_times = event_time_dict_to_mat_arrays(event_time_dict)
        restored = event_time_dict_from_mat_arrays(event_names, event_times)

        self.assertEqual(
            {"wake_nrem": [10, 20], "foot_shock_0.1s": [30.5]},
            {event: times.tolist() for event, times in restored.items()},
        )

    def test_event_time_mat_fields_survive_savemat_loadmat_round_trip(self):
        with TemporaryDirectory() as temp_dir:
            mat_path = Path(temp_dir) / "events.mat"
            mat = {"fp_frequency": np.array(1)}
            write_event_time_dict_to_mat(
                mat,
                {"wake_nrem": np.array([10, 20]), "nrem_rem": np.array([30])},
            )

            savemat(mat_path, mat)
            restored_mat = loadmat(mat_path, squeeze_me=True)
            restored = event_time_dict_from_mat(restored_mat)

        self.assertTrue(mat_has_saved_event_time_dict(restored_mat))
        self.assertIn(EVENT_TIME_NAMES_MAT_FIELD, restored_mat)
        self.assertIn(EVENT_TIME_VALUES_MAT_FIELD, restored_mat)
        self.assertEqual(
            {"wake_nrem": [10, 20], "nrem_rem": [30]},
            {event: times.tolist() for event, times in restored.items()},
        )

    def test_empty_saved_event_time_mat_fields_override_legacy_events(self):
        with TemporaryDirectory() as temp_dir:
            mat_path = Path(temp_dir) / "events.mat"
            mat = {}
            write_event_time_dict_to_mat(mat, {})

            savemat(mat_path, mat)
            restored_mat = loadmat(mat_path, squeeze_me=True)
            restored = event_time_dict_from_mat(restored_mat)

        self.assertTrue(mat_has_saved_event_time_dict(restored_mat))
        self.assertEqual({}, restored)


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


class TestEventBoundaryFiltering(unittest.TestCase):
    def test_filters_event_times_that_exceed_available_signal_samples(self):
        fp_freq = 1017.2526245117188
        signal_length = 15496205
        duration = 15234
        event_utils = Event_Utils(
            fp_freq=fp_freq,
            duration=duration,
            nsec_before=30,
            nsec_after=60,
            signal_length=signal_length,
        )
        df_events = pd.DataFrame({"sws_MA": [15173, 15174, 15202]})

        events = event_utils.read_events(df_events=df_events)

        self.assertEqual({"sws_MA": [15173]}, {k: v.tolist() for k, v in events.items()})
        perievent_windows = event_utils.make_perievent_windows(events["sws_MA"])
        perievent_indices = event_utils.get_perievent_indices(perievent_windows)
        self.assertLess(int(perievent_indices.max()), signal_length)


class TestMakeFigureFallbacks(unittest.TestCase):
    def test_embedded_event_detection_rejects_missing_and_empty_payloads(self):
        self.assertFalse(has_embedded_event_data(None))
        self.assertFalse(has_embedded_event_data(np.array([])))
        self.assertTrue(
            has_embedded_event_data(
                np.array(
                    [
                        "event",
                        np.array([[10.0, 11.0]]),
                    ],
                    dtype=object,
                )
            )
        )

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


class TestSelectiveAnalysisWorkbookExport(unittest.TestCase):
    def _build_export_payload(self, mat_filepath, auc_values=None, decay_values=None):
        auc_values = auc_values or [1.0, 2.0]
        decay_values = decay_values or [3.0, 4.0]
        return {
            "mat_filepath": mat_filepath,
            "subject_id": "F268",
            "selected_signals": ("NE2m",),
            "baseline_window": 30,
            "analysis_window": 60,
            "event_names": ["wake_sws"],
            "signal_event_exports": {
                "mean_trace": {
                    "NE2m": {
                        "wake_sws": pd.DataFrame(
                            {
                                "time_s": [0.0, 0.1],
                                "F268_mean": [0.5, 0.6],
                                "F268_sd": [0.05, 0.06],
                                "F268_n": [2, 2],
                            }
                        )
                    }
                },
                "auc": {
                    "NE2m": {
                        "wake_sws": pd.DataFrame(
                            {
                                "event_index": [1, 2],
                                "F268": auc_values,
                            }
                        )
                    }
                },
                "decay_time": {
                    "NE2m": {
                        "wake_sws": pd.DataFrame(
                            {
                                "event_index": [1, 2],
                                "F268": decay_values,
                            }
                        )
                    }
                },
            },
            "cross_correlation_event_exports": {},
            "strongest_cross_correlation_event_exports": {},
        }

    def test_selective_export_creates_only_selected_workbooks(self):
        with TemporaryDirectory() as tmpdir:
            export_payload = self._build_export_payload(Path("C:/data/F268.mat"))
            write_analysis_workbooks(
                primary_dir=Path(tmpdir),
                fallback_dir=Path(tmpdir) / "fallback",
                export_payload=export_payload,
                selected_analysis_types=["auc"],
            )
            export_dir = Path(tmpdir) / "NE2m_bw30_aw60"

            self.assertTrue((export_dir / "NE2m_auc_bw30_aw60.xlsx").exists())
            self.assertFalse((export_dir / "NE2m_bw30_aw60.xlsx").exists())
            self.assertFalse((export_dir / "NE2m_decay_time_bw30_aw60.xlsx").exists())

    def test_later_export_adds_new_analysis_type_and_updates_description(self):
        with TemporaryDirectory() as tmpdir:
            export_payload = self._build_export_payload(Path("C:/data/F268.mat"))
            write_analysis_workbooks(
                primary_dir=Path(tmpdir),
                fallback_dir=Path(tmpdir) / "fallback",
                export_payload=export_payload,
                selected_analysis_types=["auc"],
            )
            write_analysis_workbooks(
                primary_dir=Path(tmpdir),
                fallback_dir=Path(tmpdir) / "fallback",
                export_payload=export_payload,
                selected_analysis_types=["decay_time"],
            )
            export_dir = Path(tmpdir) / "NE2m_bw30_aw60"
            description_text = (export_dir / "data_description.txt").read_text(
                encoding="utf-8"
            )

            self.assertTrue((export_dir / "NE2m_auc_bw30_aw60.xlsx").exists())
            self.assertTrue((export_dir / "NE2m_decay_time_bw30_aw60.xlsx").exists())
            self.assertIn("Saved analysis types: AUC, Decay time", description_text)

    def test_resaving_same_subject_replaces_existing_subject_column(self):
        with TemporaryDirectory() as tmpdir:
            export_payload = self._build_export_payload(
                Path("C:/data/F268.mat"),
                auc_values=[1.0, 2.0],
            )
            replacement_payload = self._build_export_payload(
                Path("C:/data/F268.mat"),
                auc_values=[9.0, 8.0],
            )
            write_analysis_workbooks(
                primary_dir=Path(tmpdir),
                fallback_dir=Path(tmpdir) / "fallback",
                export_payload=export_payload,
                selected_analysis_types=["auc"],
            )
            write_analysis_workbooks(
                primary_dir=Path(tmpdir),
                fallback_dir=Path(tmpdir) / "fallback",
                export_payload=replacement_payload,
                selected_analysis_types=["auc"],
            )
            workbook_path = Path(tmpdir) / "NE2m_bw30_aw60" / "NE2m_auc_bw30_aw60.xlsx"
            exported = pd.read_excel(
                workbook_path,
                sheet_name="wake_sws",
                engine="openpyxl",
            )

        self.assertEqual(["event_index", "F268"], exported.columns.tolist())
        np.testing.assert_allclose(exported["F268"].to_numpy(), np.array([9.0, 8.0]))

    def test_checklist_values_reuse_available_remembered_analysis_types(self):
        options = [
            {"label": "Mean trace", "value": "mean_trace"},
            {"label": "AUC", "value": "auc"},
            {"label": "Mean cross-correlation", "value": "cross_correlation"},
        ]

        values = get_analysis_type_checklist_values(
            options=options,
            remembered_analysis_types=["auc", "cross_correlation"],
        )

        self.assertEqual(["auc", "cross_correlation"], values)

    def test_checklist_values_ignore_unavailable_remembered_analysis_types(self):
        options = [
            {"label": "Mean trace", "value": "mean_trace"},
            {"label": "AUC", "value": "auc"},
        ]

        values = get_analysis_type_checklist_values(
            options=options,
            remembered_analysis_types=["cross_correlation"],
        )

        self.assertEqual(["mean_trace", "auc"], values)

if __name__ == "__main__":
    unittest.main()
