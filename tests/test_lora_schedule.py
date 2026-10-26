import pytest

from utils.lora_schedule import (
    DEFAULT_LORA_SCHEDULE,
    format_lora_schedule_report,
    parse_lora_schedule,
    schedule_strength,
    sigma_step_positions,
    sigma_to_step_percent,
)


def test_parse_lines_commas_percent_signs_and_negative_strengths():
    assert parse_lora_schedule("\n 0%: -1, 25:0.5\n100:2\n") == (
        (0, -1), (25, 0.5), (100, 2),
    )
    assert DEFAULT_LORA_SCHEDULE == "0%:1\n10%:1\n40%:0\n100%:0"
    assert parse_lora_schedule(DEFAULT_LORA_SCHEDULE) == parse_lora_schedule("0:1,10:1,40:0,100:0")


@pytest.mark.parametrize("text", [
    "", " , \n", "0=1", "0:1:2", "cat:1", "0:dog",
    "-1:1", "101:1", "nan:1", "inf:1", "0:nan", "0:inf",
    "0:101", "0:-101", "30:1, 20:0", "0:1, 0:0",
])
def test_invalid_schedules_raise_clear_errors(text):
    with pytest.raises(ValueError, match="LoRA schedule"):
        parse_lora_schedule(text)


@pytest.mark.parametrize(("percent", "expected"), [
    (0, 1), (5, 1), (10, 1), (25, 0.5), (40, 0), (90, 0), (100, 0),
])
def test_default_hold_fade_off(percent, expected):
    assert schedule_strength(parse_lora_schedule(DEFAULT_LORA_SCHEDULE), percent) == pytest.approx(expected)


def test_arbitrary_fade_in_hold_fade_out_and_endpoint_extension():
    points = parse_lora_schedule("10:0, 20:1, 60:1, 90:-1")
    for percent, expected in [(0, 0), (15, 0.5), (40, 1), (75, 0), (100, -1)]:
        assert schedule_strength(points, percent) == pytest.approx(expected)


def test_constant_switches_at_exact_points_and_single_point_holds():
    points = parse_lora_schedule("0:1, 30:0, 50:-1")
    assert [schedule_strength(points, p, "constant") for p in [0, 29, 30, 49, 50, 100]] == [1, 1, 0, 0, -1, -1]
    assert schedule_strength(parse_lora_schedule("25:0.7"), 0) == 0.7
    assert schedule_strength(parse_lora_schedule("25:0.7"), 100) == 0.7
    with pytest.raises(ValueError, match="interpolation"):
        schedule_strength(points, 0, "unknown")


def test_progress_uses_step_intervals_instead_of_noise_percentage():
    positions = sigma_step_positions([100, 10, 1, 0.1, 0])
    assert [sigma_to_step_percent(s, positions) for s in [100, 10, 1, 0.1, 0]] == [0, 25, 50, 75, 100]
    assert sigma_to_step_percent(5.5, positions) == 37.5
    assert sigma_to_step_percent(200, positions) == 0
    # Repeated or out-of-order model evaluations do not advance a counter.
    assert sigma_to_step_percent(1, positions) == 50
    assert sigma_to_step_percent(10, positions) == 25


def test_reduced_denoise_schedule_starts_at_zero_and_default_hits_step_eight():
    sigmas = [20 - step for step in range(21)]
    points = parse_lora_schedule(DEFAULT_LORA_SCHEDULE)
    positions = sigma_step_positions(sigmas)
    assert schedule_strength(points, sigma_to_step_percent(sigmas[2], positions)) == 1
    assert schedule_strength(points, sigma_to_step_percent(sigmas[5], positions)) == 0.5
    assert schedule_strength(points, sigma_to_step_percent(sigmas[8], positions)) == 0
    assert sigma_to_step_percent(5, sigma_step_positions([5, 1, 0])) == 0


@pytest.mark.parametrize("sigmas", [[], [1], [1, 1, 0], [0, 1], [1, -1], [float("inf"), 0], [float("nan"), 0]])
def test_invalid_sigma_schedules(sigmas):
    with pytest.raises(ValueError, match="[Ss]cheduled LoRA"):
        sigma_step_positions(sigmas)


def test_nonfinite_sampling_sigma_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        sigma_to_step_percent(float("nan"), (-1, 0))


def test_report_uses_one_based_steps_and_excludes_terminal_boundary_from_step_rows():
    report = format_lora_schedule_report(parse_lora_schedule(DEFAULT_LORA_SCHEDULE), "linear", 20)
    rows = [line for line in report.splitlines() if line.startswith("Step ")]
    assert len(rows) == 20
    assert rows[0] == "Step 1/20 | 0% | strength 1"
    assert rows[2] == "Step 3/20 | 10% | strength 1"
    assert rows[5] == "Step 6/20 | 25% | strength 0.5"
    assert rows[8] == "Step 9/20 | 40% | strength 0"
    assert rows[-1] == "Step 20/20 | 95% | strength 0"
    assert "End boundary | 100% | strength 0 (usually not evaluated)" in report
    assert "not a runtime log" in report


def test_report_matches_runtime_strength_at_nonlinear_sigma_boundaries():
    points = parse_lora_schedule("0:-1, 50:1, 100:0")
    sigmas = [100, 10, 1, 0.1, 0]
    positions = sigma_step_positions(sigmas)
    report = format_lora_schedule_report(points, "linear", 4)
    rows = [line for line in report.splitlines() if line.startswith("Step ")]
    for sigma, row in zip(sigmas[:-1], rows):
        expected = schedule_strength(points, sigma_to_step_percent(sigma, positions))
        assert float(row.rsplit(" ", 1)[1]) == pytest.approx(expected)


def test_report_constant_mode_single_step_and_nonzero_terminal_value():
    points = parse_lora_schedule("0:1, 50:-0.5, 100:2")
    report = format_lora_schedule_report(points, "constant", 4)
    assert "Step 2/4 | 25% | strength 1" in report
    assert "Step 3/4 | 50% | strength -0.5" in report
    assert "Step 4/4 | 75% | strength -0.5" in report
    assert "End boundary | 100% | strength 2" in report
    single = format_lora_schedule_report(points, "linear", 1)
    assert "Step 1/1 | 0% | strength 1" in single
    assert "Step 2/" not in single


@pytest.mark.parametrize("steps", [0, -1, 10001, 2.5, True, "20", None])
def test_report_rejects_invalid_step_counts(steps):
    with pytest.raises(ValueError, match="report_steps"):
        format_lora_schedule_report(((0, 1),), "linear", steps)
