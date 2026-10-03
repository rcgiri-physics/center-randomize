from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import json
import math
import sys

# Parameters - keep in sync with school_center.py (not imported from there because
# school_center.py parses command line arguments and allocates at import time)
PREF_DISTANCE_THRESHOLD = 2  # Preferred threshold distance in km
ABS_DISTANCE_THRESHOLD = 7  # Absolute threshold distance in km
MIN_STUDENT_IN_CENTER = (
    10  # Min. no of students from a school to be assigned to a center
)
STRETCH_CAPACITY_FACTOR = (
    0.02  # How much can center capacity be streched if need arises
)
PREF_CUTOFF = -4  # Do not allocate students with pref score less than cutoff

ERROR = "error"  # Rule that must hold
WARNING = "warning"  # Rule the guidelines say to follow "as far as possible"

# Rule ids, numbered as in README.md "Allocation Guidelines" / the task description
R_CAPACITY = "capacity"  # 1
R_DISTANCE = "distance"  # 2 (hard limit)
R_SELF_CENTER = "self-center"  # 3
R_MUTUAL = "mutual-centers"  # 4
R_FORBIDDEN_PAIR = "forbidden-pair"  # 5 and 6 (prefs score <= PREF_CUTOFF)
R_MAX_PER_CENTER = "max-per-center"  # 7
R_MIN_PER_CENTER = "min-per-center"  # 8
R_RANDOM = "randomness"  # 9
R_UNALLOCATED = "unallocated"  # every student needs a seat
R_OVERALLOCATED = "overallocated"
R_INTEGRITY = "integrity"  # unknown codes, duplicate rows, bad data


@dataclass
class Violation:
    rule: str
    severity: str
    message: str
    scode: Optional[str] = None
    cscode: Optional[str] = None


@dataclass
class Report:
    violations: List[Violation] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == ERROR]

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def count_by_rule(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for v in self.violations:
            counts[v.rule] = counts.get(v.rule, 0) + 1
        return counts


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees
    - Reference: https://en.wikipedia.org/wiki/Haversine_formula
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius_earth = 6371  # Average Radius of Earth in km
    return radius_earth * c


def read_tsv(file_path: str) -> List[Dict[str, str]]:
    """Read a tab separated file into a list of dicts."""
    with open(file_path, "r", newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file, delimiter="\t")]


def read_prefs(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Read prefs.tsv into {scode: {cscode: score}}.
    Repeated pairs are summed, same as school_center.py does.
    """
    prefs: Dict[str, Dict[str, int]] = {}
    for row in read_tsv(file_path):
        pair = prefs.setdefault(row["scode"], {})
        pair[row["cscode"]] = pair.get(row["cscode"], 0) + int(row["pref"])
    return prefs


def max_per_center(student_count: int) -> int:
    """Max students of one school at one center, mirrors calc_per_center()."""
    return 100 if student_count <= 400 else 200


def _to_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate(
    schools: List[Dict[str, str]],
    centers: List[Dict[str, str]],
    prefs: Dict[str, Dict[str, int]],
    allocation: List[Dict[str, str]],
    strict: bool = False,
) -> Report:
    """
    Check an allocation (rows of results/school-center.tsv) against every rule.
    Distances are recomputed from the input coordinates; distance_km in the
    output is only cross checked. With strict=True the "as far as possible" rules
    (max students per center, min students per center) are reported as errors.
    """
    report = Report()
    soft = ERROR if strict else WARNING

    def add(rule, severity, message, scode=None, cscode=None):
        report.violations.append(Violation(rule, severity, message, scode, cscode))

    school_by_code = {s["scode"]: s for s in schools}
    center_by_code = {c["cscode"]: c for c in centers}

    # Parse and sanity check rows, merging nothing: one row per (school, center)
    pairs: Dict[Tuple[str, str], int] = {}
    row_distance: Dict[Tuple[str, str], Optional[float]] = {}
    for row in allocation:
        scode, cscode = row["scode"], row["cscode"]
        try:
            count = int(row["allocation"])
        except (TypeError, ValueError):
            add(
                R_INTEGRITY,
                ERROR,
                f"non numeric allocation {row.get('allocation')!r}",
                scode,
                cscode,
            )
            continue
        if count <= 0:
            add(
                R_INTEGRITY, ERROR, f"allocation {count} is not positive", scode, cscode
            )
            continue
        if scode not in school_by_code:
            add(R_INTEGRITY, ERROR, "school code not in schools file", scode, cscode)
            continue
        if cscode not in center_by_code:
            add(R_INTEGRITY, ERROR, "center code not in centers file", scode, cscode)
            continue
        if (scode, cscode) in pairs:
            add(
                R_INTEGRITY,
                ERROR,
                "school-center pair appears in more than one row",
                scode,
                cscode,
            )
        pairs[(scode, cscode)] = pairs.get((scode, cscode), 0) + count
        row_distance[(scode, cscode)] = _to_float(row.get("distance_km"))

    # Rule 3: school is not a center for itself
    for scode, cscode in pairs:
        if scode == cscode:
            add(
                R_SELF_CENTER,
                ERROR,
                "school is assigned to itself as center",
                scode,
                cscode,
            )

    # Rule 4: no two schools use each other's centers
    for scode, cscode in pairs:
        if scode < cscode and (cscode, scode) in pairs:
            add(
                R_MUTUAL,
                ERROR,
                f"{scode} and {cscode} are each other's centers",
                scode,
                cscode,
            )

    # Rule 5 and 6: same management / past problems both show up as prefs <= cutoff
    for scode, cscode in pairs:
        score = prefs.get(scode, {}).get(cscode, 0)
        if score <= PREF_CUTOFF:
            add(
                R_FORBIDDEN_PAIR,
                ERROR,
                f"pref score {score} <= cutoff {PREF_CUTOFF}",
                scode,
                cscode,
            )

    # Rule 2: distance, recomputed from coordinates
    allocated_students = 0
    distance_weighted = 0.0
    within_preferred = 0
    for (scode, cscode), count in pairs.items():
        s, c = school_by_code[scode], center_by_code[cscode]
        coords = [
            _to_float(s.get("lat")),
            _to_float(s.get("long")),
            _to_float(c.get("lat")),
            _to_float(c.get("long")),
        ]
        if None in coords:
            add(
                R_DISTANCE,
                ERROR,
                "distance cannot be computed, coordinates missing",
                scode,
                cscode,
            )
            continue
        distance = haversine_distance(*coords)
        allocated_students += count
        distance_weighted += distance * count
        if distance <= PREF_DISTANCE_THRESHOLD:
            within_preferred += count
        if distance > ABS_DISTANCE_THRESHOLD:
            add(
                R_DISTANCE,
                ERROR,
                f"distance {distance:.2f} km exceeds {ABS_DISTANCE_THRESHOLD} km",
                scode,
                cscode,
            )
        reported = row_distance[(scode, cscode)]
        if reported is not None and abs(reported - distance) > 0.01:
            add(
                R_INTEGRITY,
                ERROR,
                f"distance_km {reported:.3f} in output differs from computed {distance:.3f}",
                scode,
                cscode,
            )

    # Rule 1: capacity, with the allowed stretch
    per_center: Dict[str, int] = {}
    for (_, cscode), count in pairs.items():
        per_center[cscode] = per_center.get(cscode, 0) + count
    stretched_centers = 0
    for cscode, total in per_center.items():
        capacity = int(center_by_code[cscode]["capacity"])
        limit = capacity + math.floor(capacity * STRETCH_CAPACITY_FACTOR)
        if total > limit:
            add(
                R_CAPACITY,
                ERROR,
                f"{total} students exceed capacity {capacity} "
                f"(limit with stretch {limit})",
                cscode=cscode,
            )
        elif total > capacity:
            stretched_centers += 1

    # Rule 7 and 8, per school-center pair
    for (scode, cscode), count in pairs.items():
        school_count = int(school_by_code[scode]["count"])
        limit = max_per_center(school_count)
        if count > limit:
            add(
                R_MAX_PER_CENTER,
                soft,
                f"{count} students at one center, guideline is {limit}",
                scode,
                cscode,
            )
        # A school smaller than the minimum can only ever have a small assignment
        if count < MIN_STUDENT_IN_CENTER and school_count >= MIN_STUDENT_IN_CENTER:
            add(
                R_MIN_PER_CENTER,
                soft,
                f"only {count} students at center, minimum is {MIN_STUDENT_IN_CENTER}",
                scode,
                cscode,
            )

    # Every student must get a seat, and none may be counted twice
    per_school: Dict[str, int] = {}
    for (scode, _), count in pairs.items():
        per_school[scode] = per_school.get(scode, 0) + count
    unallocated = 0
    for scode, s in school_by_code.items():
        wanted = int(s["count"])
        got = per_school.get(scode, 0)
        if got < wanted:
            unallocated += wanted - got
            add(
                R_UNALLOCATED,
                ERROR,
                f"{wanted - got} of {wanted} students have no center",
                scode,
            )
        elif got > wanted:
            add(
                R_OVERALLOCATED,
                ERROR,
                f"{got} students allocated but school has {wanted}",
                scode,
            )

    total_students = sum(int(s["count"]) for s in schools)
    report.stats = {
        "total_students": total_students,
        "allocated_students": allocated_students,
        "unallocated_students": unallocated,
        "centers_used": len(per_center),
        "centers_stretched": stretched_centers,
        "avg_distance_km": distance_weighted / allocated_students
        if allocated_students
        else 0.0,
        "share_within_preferred": within_preferred / allocated_students
        if allocated_students
        else 0.0,
    }
    return report


def validate_randomness(allocations: List[List[Dict[str, str]]]) -> List[Violation]:
    """
    Rule 9: runs with different seeds must not produce identical allocations.
    Pass the rows of two or more outputs; every identical pair is reported.
    """

    def fingerprint(rows):
        return sorted((r["scode"], r["cscode"], int(r["allocation"])) for r in rows)

    prints = [fingerprint(rows) for rows in allocations]
    violations = []
    for i in range(len(prints)):
        for j in range(i + 1, len(prints)):
            if prints[i] == prints[j]:
                violations.append(
                    Violation(
                        R_RANDOM, ERROR, f"outputs {i + 1} and {j + 1} are identical"
                    )
                )
    return violations


def format_report(report: Report, max_examples: int = 5) -> str:
    """Human readable summary: stats, then counts and a few examples per rule."""
    lines = []
    s = report.stats
    lines.append(
        f"Students: {s['total_students']} total, {s['allocated_students']} allocated, "
        f"{s['unallocated_students']} unallocated"
    )
    lines.append(
        f"Centers used: {s['centers_used']} ({s['centers_stretched']} over nominal capacity)"
    )
    lines.append(
        f"Average distance: {s['avg_distance_km']:.2f} km, "
        f"within {PREF_DISTANCE_THRESHOLD} km: {s['share_within_preferred']:.1%}"
    )
    by_rule: Dict[str, List[Violation]] = {}
    for v in report.violations:
        by_rule.setdefault(v.rule, []).append(v)
    if not by_rule:
        lines.append("No violations found.")
    for rule, items in sorted(by_rule.items()):
        lines.append(f"[{items[0].severity.upper()}] {rule}: {len(items)}")
        for v in items[:max_examples]:
            where = " ".join(
                p
                for p in (
                    f"school {v.scode}" if v.scode else "",
                    f"center {v.cscode}" if v.cscode else "",
                )
                if p
            )
            lines.append(f"    {where}: {v.message}" if where else f"    {v.message}")
        if len(items) > max_examples:
            lines.append(f"    ... {len(items) - max_examples} more")
    lines.append("RESULT: " + ("PASS" if report.ok else "FAIL"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate allocation",
        description="Check a school-center allocation against the allocation rules",
    )
    parser.add_argument(
        "schools_tsv", help="Tab separated (TSV) file containing school details"
    )
    parser.add_argument(
        "centers_tsv", help="Tab separated (TSV) file containing center details"
    )
    parser.add_argument(
        "prefs_tsv", help="Tab separated (TSV) file containing preference scores"
    )
    parser.add_argument(
        "allocation_tsv", help="Allocation output (school-center.tsv) to check"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat the 'as far as possible' rules (max/min students per center) as errors",
    )
    parser.add_argument(
        "--compare",
        metavar="OTHER_TSV",
        nargs="+",
        default=[],
        help="Other outputs of the same input, checked to differ from the first (rule 9)",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)

    allocation = read_tsv(args.allocation_tsv)
    report = validate(
        read_tsv(args.schools_tsv),
        read_tsv(args.centers_tsv),
        read_prefs(args.prefs_tsv),
        allocation,
        strict=args.strict,
    )
    if args.compare:
        report.violations += validate_randomness(
            [allocation] + [read_tsv(path) for path in args.compare]
        )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "stats": report.stats,
                    "counts": report.count_by_rule(),
                    "violations": [v.__dict__ for v in report.violations],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
