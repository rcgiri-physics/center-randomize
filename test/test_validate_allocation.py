import csv
import json
import subprocess
import sys

import pytest

import validate_allocation as va
from validate_allocation import validate, validate_randomness

KM_PER_DEGREE_LAT = 111.19  # 2 * pi * 6371 / 360


def school(scode, count=50, km=0.0):
    """School placed km north of the origin, so distances are easy to reason about."""
    return {
        "scode": scode,
        "count": str(count),
        "name-address": f"school {scode}",
        "lat": str(27 + km / KM_PER_DEGREE_LAT),
        "long": "85.3",
    }


def center(cscode, capacity=500, km=0.0):
    return {
        "cscode": cscode,
        "capacity": str(capacity),
        "name": f"center {cscode}",
        "lat": str(27 + km / KM_PER_DEGREE_LAT),
        "long": "85.3",
    }


def row(scode, cscode, allocation):
    return {"scode": scode, "cscode": cscode, "allocation": str(allocation)}


def rules(report, severity=None):
    return {v.rule for v in report.violations if severity in (None, v.severity)}


def test_clean_allocation_passes_and_reports_stats():
    schools = [school("1", 60, km=0), school("2", 40, km=0)]
    centers = [center("10", km=1), center("11", km=4)]
    allocation = [row("1", "10", 60), row("2", "11", 40)]
    report = validate(schools, centers, {}, allocation)
    assert report.ok
    assert report.violations == []
    assert report.stats["total_students"] == 100
    assert report.stats["unallocated_students"] == 0
    assert report.stats["share_within_preferred"] == pytest.approx(0.6)
    assert report.stats["avg_distance_km"] == pytest.approx(0.6 * 1 + 0.4 * 4, abs=0.01)


def test_capacity_stretch_is_allowed_but_not_beyond():
    schools = [school("1", 100), school("2", 100)]
    # capacity 100 -> stretched limit 102
    ok = validate(schools[:1], [center("10", capacity=100)], {}, [row("1", "10", 100)])
    assert ok.ok and ok.stats["centers_stretched"] == 0
    stretched = validate(
        [school("1", 52), school("2", 50)],
        [center("10", capacity=100)],
        {},
        [row("1", "10", 52), row("2", "10", 50)],
    )
    assert stretched.ok and stretched.stats["centers_stretched"] == 1
    over = validate(
        [school("1", 53), school("2", 50)],
        [center("10", capacity=100)],
        {},
        [row("1", "10", 53), row("2", "10", 50)],
    )
    assert rules(over, va.ERROR) == {va.R_CAPACITY}


def test_distance_hard_limit():
    centers = [center("10", km=1.9), center("11", km=6.9), center("12", km=7.5)]
    schools = [school("1", 30), school("2", 30), school("3", 30)]
    report = validate(
        schools,
        centers,
        {},
        [row("1", "10", 30), row("2", "11", 30), row("3", "12", 30)],
    )
    assert [v.scode for v in report.violations if v.rule == va.R_DISTANCE] == ["3"]
    # 6.9 km is allowed but is not within the preferred distance
    assert report.stats["share_within_preferred"] == pytest.approx(1 / 3)


def test_missing_coordinates_cannot_be_checked():
    s = school("1", 30)
    s["lat"] = ""
    report = validate([s], [center("10")], {}, [row("1", "10", 30)])
    assert rules(report, va.ERROR) == {va.R_DISTANCE}


def test_school_cannot_be_its_own_center():
    report = validate([school("1", 30)], [center("1")], {}, [row("1", "1", 30)])
    assert va.R_SELF_CENTER in rules(report, va.ERROR)


def test_mutual_centers_are_reported_once():
    schools = [school("1", 30), school("2", 30)]
    centers = [center("1"), center("2")]
    report = validate(schools, centers, {}, [row("1", "2", 30), row("2", "1", 30)])
    mutual = [v for v in report.violations if v.rule == va.R_MUTUAL]
    assert len(mutual) == 1


def test_one_way_assignment_is_not_mutual():
    schools = [school("1", 30), school("2", 30), school("3", 30)]
    centers = [center("1"), center("2"), center("3")]
    report = validate(
        schools, centers, {}, [row("1", "2", 30), row("2", "3", 30), row("3", "1", 30)]
    )
    assert va.R_MUTUAL not in rules(report)


@pytest.mark.parametrize(
    "score, violates", [(-5, True), (-4, True), (-3, False), (0, False), (5, False)]
)
def test_prefs_cutoff(score, violates):
    prefs = {"1": {"10": score}}
    report = validate([school("1", 30)], [center("10")], prefs, [row("1", "10", 30)])
    assert (va.R_FORBIDDEN_PAIR in rules(report, va.ERROR)) is violates


def test_prefs_are_directional():
    prefs = {
        "10": {"1": -5}
    }  # centre 10's own students must not go to 1, says nothing about 1 -> 10
    report = validate([school("1", 30)], [center("10")], prefs, [row("1", "10", 30)])
    assert va.R_FORBIDDEN_PAIR not in rules(report)


@pytest.mark.parametrize("students, per_center_limit", [(400, 100), (401, 200)])
def test_max_students_per_center_limit_depends_on_school_size(
    students, per_center_limit
):
    centers = [center("10", capacity=1000)]
    at_limit = validate(
        [school("1", students)], centers, {}, [row("1", "10", per_center_limit)]
    )
    assert va.R_MAX_PER_CENTER not in rules(at_limit)
    above = validate(
        [school("1", students)], centers, {}, [row("1", "10", per_center_limit + 1)]
    )
    assert va.R_MAX_PER_CENTER in rules(above)


def test_soft_rules_are_warnings_unless_strict():
    schools = [school("1", 150), school("2", 55)]
    centers = [center("10", capacity=1000), center("11", capacity=1000)]
    allocation = [row("1", "10", 150), row("2", "10", 50), row("2", "11", 5)]
    relaxed = validate(schools, centers, {}, allocation)
    assert relaxed.ok
    assert rules(relaxed, va.WARNING) == {va.R_MAX_PER_CENTER, va.R_MIN_PER_CENTER}
    strict = validate(schools, centers, {}, allocation, strict=True)
    assert not strict.ok
    assert rules(strict, va.ERROR) == {va.R_MAX_PER_CENTER, va.R_MIN_PER_CENTER}


def test_school_smaller_than_minimum_is_not_flagged():
    report = validate([school("1", 6)], [center("10")], {}, [row("1", "10", 6)])
    assert report.violations == []


def test_unallocated_students_are_an_error():
    report = validate(
        [school("1", 50), school("2", 20)], [center("10")], {}, [row("1", "10", 45)]
    )
    unallocated = [v for v in report.violations if v.rule == va.R_UNALLOCATED]
    assert {v.scode for v in unallocated} == {"1", "2"}
    assert report.stats["unallocated_students"] == 25
    assert not report.ok


def test_overallocated_school_is_an_error():
    report = validate([school("1", 50)], [center("10")], {}, [row("1", "10", 51)])
    assert rules(report, va.ERROR) == {va.R_OVERALLOCATED}


@pytest.mark.parametrize(
    "bad_row",
    [
        row("99", "10", 10),  # school not in the schools file
        row("1", "99", 10),  # center not in the centers file
        row("1", "10", 0),
        row("1", "10", -3),
        row("1", "10", "many"),
    ],
)
def test_integrity_problems_are_errors(bad_row):
    report = validate([school("1", 50)], [center("10")], {}, [bad_row])
    assert va.R_INTEGRITY in rules(report, va.ERROR)


def test_duplicate_pair_rows_are_an_error():
    report = validate(
        [school("1", 50)], [center("10")], {}, [row("1", "10", 25), row("1", "10", 25)]
    )
    assert va.R_INTEGRITY in rules(report, va.ERROR)


def test_reported_distance_is_cross_checked():
    good = dict(row("1", "10", 30), distance_km="3.0")
    bad = dict(row("1", "10", 30), distance_km="1.0")
    schools, centers = [school("1", 30)], [center("10", km=3)]
    assert validate(schools, centers, {}, [good]).violations == []
    assert va.R_INTEGRITY in rules(validate(schools, centers, {}, [bad]), va.ERROR)


def test_randomness_detects_identical_outputs():
    a = [row("1", "10", 30), row("2", "11", 20)]
    same_in_other_order = [row("2", "11", 20), row("1", "10", 30)]
    different = [row("1", "11", 30), row("2", "10", 20)]
    assert validate_randomness([a, different]) == []
    identical = validate_randomness([a, same_in_other_order, different])
    assert len(identical) == 1 and identical[0].rule == va.R_RANDOM


def test_read_prefs_sums_repeated_pairs(tmp_path):
    path = tmp_path / "prefs.tsv"
    path.write_text(
        "scode\tcscode\tpref\treason\n1\t2\t-2\ta\n1\t2\t-3\tb\n1\t3\t1\tc\n",
        encoding="utf-8",
    )
    assert va.read_prefs(str(path)) == {"1": {"2": -5, "3": 1}}


def write_tsv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def tsv_inputs(tmp_path):
    schools = write_tsv(
        tmp_path / "schools.tsv",
        [school("1", 30), school("2", 20)],
        ["scode", "count", "name-address", "lat", "long"],
    )
    centers = write_tsv(
        tmp_path / "centers.tsv",
        [center("10"), center("11")],
        ["cscode", "capacity", "name", "lat", "long"],
    )
    prefs = tmp_path / "prefs.tsv"
    prefs.write_text("scode\tcscode\tpref\treason\n", encoding="utf-8")
    return str(schools), str(centers), str(prefs)


def test_cli_exit_codes_and_json(tmp_path, tsv_inputs, capsys):
    fields = ["scode", "cscode", "allocation"]
    good = write_tsv(
        tmp_path / "good.tsv", [row("1", "10", 30), row("2", "11", 20)], fields
    )
    bad = write_tsv(tmp_path / "bad.tsv", [row("1", "10", 30)], fields)
    assert va.main([*tsv_inputs, good]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    assert va.main([*tsv_inputs, bad]) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out
    assert va.main([*tsv_inputs, bad, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["counts"] == {va.R_UNALLOCATED: 1}


def test_cli_compare_flags_identical_runs(tmp_path, tsv_inputs):
    fields = ["scode", "cscode", "allocation"]
    a = write_tsv(tmp_path / "a.tsv", [row("1", "10", 30), row("2", "11", 20)], fields)
    b = write_tsv(tmp_path / "b.tsv", [row("1", "11", 30), row("2", "10", 20)], fields)
    assert va.main([*tsv_inputs, a, "--compare", b]) == 0
    assert va.main([*tsv_inputs, a, "--compare", a]) == 1


def test_existing_allocator_honours_the_structural_rules(tmp_path):
    """
    Runs school_center.py on sample_data and checks its output. Only rules the
    current algorithm guarantees are asserted - it can still leave students
    unallocated, assign a center beyond 7 km and break the soft rules, see the
    validator report for those.
    """
    out = tmp_path / "school-center.tsv"
    subprocess.run(
        [
            sys.executable,
            "school_center.py",
            "sample_data/schools_grade12_2081.tsv",
            "sample_data/centers_grade12_2081.tsv",
            "sample_data/prefs.tsv",
            "-o",
            str(out),
            "-s",
            "7",
        ],
        check=True,
        capture_output=True,
    )
    report = validate(
        va.read_tsv("sample_data/schools_grade12_2081.tsv"),
        va.read_tsv("sample_data/centers_grade12_2081.tsv"),
        va.read_prefs("sample_data/prefs.tsv"),
        va.read_tsv(str(out)),
    )
    structural = {
        va.R_CAPACITY,
        va.R_SELF_CENTER,
        va.R_MUTUAL,
        va.R_FORBIDDEN_PAIR,
        va.R_OVERALLOCATED,
        va.R_INTEGRITY,
    }
    assert not rules(report) & structural
