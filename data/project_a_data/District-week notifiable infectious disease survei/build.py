import csv, re, os
from collections import OrderedDict, defaultdict

RAW = "/tmp/idsr/raw/tables.txt"
OUT = "/tmp/idsr/extracted"
os.makedirs(OUT, exist_ok=True)

# ---- canonical condition names (harmonise header spelling variants) ----
def canon(h):
    k = (h.strip().lower()
         .replace(" ", "")
         .replace(".", "")
         .replace("<", "")
         .replace("/", "")
         .replace("-", ""))
    m = {
        # Acute Diarrhoea (non-cholera) -- incl. two one-off transcription
        # artifacts ("NonCholeraA" / "NonCholeraI", 2025 w12/w13) that are the
        # same column in the same position, almost certainly a stray footnote
        # marker picked up from the source PDF.
        "ad(noncholera)": "AD_noncholera",
        "ad(noncholeraa)": "AD_noncholera",
        "ad(noncholerai)": "AD_noncholera",
        "malaria": "Malaria",
        "ili": "ILI",
        "sari": "SARI",
        "alri5years": "ALRI_u5", "alri5year": "ALRI_u5", "alri5": "ALRI_u5",
        "bdiarrhea": "Bloody_diarrhoea",
        "typhoid": "Typhoid",
        "cl": "Cut_leishmaniasis",
        "avh(a&e)": "AVH",
        "dogbite": "Dog_bite", "animaldogbite": "Dog_bite",
        "measles": "Measles", "measels": "Measles",
        "tb": "TB",
        "dengue": "Dengue",
        # conditions only seen once the full 164-bulletin corpus was pulled in
        "awd(scholera)": "AWD_cholera",          # Acute Watery Diarrhoea, suspected cholera
        "vh(b,c&d)": "VH_BCD",                    # Viral Hepatitis B, C & D
        "chickenpoxvaricella": "Chickenpox",
    }
    return m.get(k, "??:"+h.strip())

# ---- district / reporting-unit crosswalk (the "boundary trap") ----
# raw label (as printed) -> (canonical_unit, unit_type, parent_district)
# unit_type: "district" | "subdivision" | "typo"
# Sub-divisions are a FINER reporting geography than the district they sit
# inside (they show up in weeks where NIH reported at sub-division level
# instead of, or in addition to, the district level) — they are intentionally
# NOT merged into the parent district, only cross-referenced, because the two
# geographies are not equivalent and silently merging them would double-count
# or mis-attribute cases.
DISTRICT_XWALK = {
    "Abbottabad": ("Abbottabad", "district", None),
    "Abbottaba": ("Abbottabad", "typo", None),          # truncated transcription
    "Bajaur": ("Bajaur", "district", None),
    "Bannu": ("Bannu", "district", None),
    "SD Bannu": ("SD Bannu", "subdivision", "Bannu"),
    "Battagram": ("Battagram", "district", None),
    "Buner": ("Buner", "district", None),
    "Charsadda": ("Charsadda", "district", None),
    "Chitral Lower": ("Chitral Lower", "district", None),
    "Chitral Upper": ("Chitral Upper", "district", None),
    "D.I. Khan": ("D.I. Khan", "district", None),
    "SD DI Khan": ("SD DI Khan", "subdivision", "D.I. Khan"),
    "Dir Lower": ("Dir Lower", "district", None),
    "Dir Upper": ("Dir Upper", "district", None),
    "Hangu": ("Hangu", "district", None),
    "Hangue": ("Hangu", "typo", None),
    "SD Kohat": ("SD Kohat", "subdivision", "Kohat"),
    "Haripur": ("Haripur", "district", None),
    "Karak": ("Karak", "district", None),
    "Khyber": ("Khyber", "district", None),
    "Kohat": ("Kohat", "district", None),
    "Kohistan Lower": ("Kohistan Lower", "district", None),
    "Kohistan Upper": ("Kohistan Upper", "district", None),
    "Kolai Palas": ("Kolai Palas", "district", None),
    "L & C Kurram": ("Lower & Central Kurram", "district", None),
    "Upper Kurram": ("Upper Kurram", "district", None),
    "Lakki Marwat": ("Lakki Marwat", "district", None),
    "SD Lakki": ("SD Lakki", "subdivision", "Lakki Marwat"),
    "Malakand": ("Malakand", "district", None),
    "Mansehra": ("Mansehra", "district", None),
    "Mardan": ("Mardan", "district", None),
    "Mohmand": ("Mohmand", "district", None),
    "North Waziristan": ("North Waziristan", "district", None),
    "NWA": ("North Waziristan", "district", None),   # North Waziristan Agency, legacy abbrev.
    "Nowshera": ("Nowshera", "district", None),
    "Orakzai": ("Orakzai", "district", None),
    "Peshawar": ("Peshawar", "district", None),
    "SD Peshawar": ("SD Peshawar", "subdivision", "Peshawar"),
    "Shangla": ("Shangla", "district", None),
    # South Waziristan: see SWA_CUTOVER below. "SWA" is TIME-DEPENDENT and is
    # deliberately NOT given a static entry here — resolving it requires the
    # bulletin week (see canon_district).
    "South Waziristan (Lower)": ("South Waziristan Lower", "district", None),
    "South Waziristan (Upper)": ("South Waziristan Upper", "district", None),
    "SWU": ("South Waziristan Upper", "district", None),
    "SW (L)": ("South Waziristan Lower", "district", None),
    "SW(U)": ("South Waziristan Upper", "district", None),
    "Swabi": ("Swabi", "district", None),
    "Swat": ("Swat", "district", None),
    "Tank": ("Tank", "district", None),
    "SD Tank": ("SD Tank", "subdivision", "Tank"),
    "Tor Ghar": ("Tor Ghar", "district", None),
}

# ---- TIME-DEPENDENT reporting unit: "SWA" ----------------------------------
# The label "SWA" (legacy "South Waziristan Agency") does not denote a fixed
# geography across the corpus. Verified empirically from the panel itself:
#
#   2023 w34 - 2024 w49 : "SWA" is the ONLY South Waziristan row present, and
#                         its all-condition weekly total runs ~350-450, against
#                         a like-for-like (same epi-weeks, 2025) Upper+Lower sum
#                         of ~548 and Upper-alone of ~158. SWA is therefore the
#                         UNDIVIDED district, not either half.
#   2024 w50 onward     : "South Waziristan (Lower)" begins reporting as its own
#                         row (~286-342/wk) and SWA collapses to ~49-61/wk. SWA
#                         is now the residual, i.e. UPPER.
#   2025 w1 onward      : the label is replaced by "SWU"; the series continues
#                         seamlessly (2024 w52 SWA = 61 -> 2025 w1 SWU = 50).
#
# The previous build mapped SWA -> "South Waziristan Lower" unconditionally.
# That was wrong in every week: it attributed undivided-district totals to Lower
# for 67 weeks, and in 2024 w49-w52 it collided with the genuine Lower row,
# producing 80 duplicate canonical keys carrying conflicting values.
SWA_CUTOVER = (2024, 50)   # first week in which "SWA" denotes Upper, not the whole district

# 2024 w49 is a genuine source-side ambiguity: "SWA" still reports at
# whole-district level (340) while a small "South Waziristan (Lower)" row (42)
# also appears. The two overlap geographically in that week. SWA is treated as
# undivided (matching its magnitude) and the week is flagged, not silently
# resolved.
SWA_AMBIGUOUS_WEEKS = [(2024, 49)]

def canon_district(raw, year=None, week=None):
    d = raw.strip()
    if d == "SWA":
        if year is None:
            raise ValueError("SWA is time-dependent; year/week required")
        if (year, week) >= SWA_CUTOVER:
            return ("South Waziristan Upper", "district", None)
        return ("South Waziristan (undivided)", "district", None)
    hit = DISTRICT_XWALK.get(d)
    if hit:
        return hit
    return (d, "UNMAPPED", None)

# ---- dropped-cell corrections ---------------------------------------------
# Some district rows in the source PDFs contain a genuinely blank cell. The
# document-reading tool returns only the populated values, left-aligned, so every
# value after the gap is transcribed one column to the LEFT of the heading it
# belongs under, and a spurious value is padded onto the end of the row. The
# digits are read correctly; only their column assignment is wrong.
#
# Such a row is undetectable by inspection — it looks complete and plausible.
# It is identified here by testing every non-reconciling row against a
# one-parameter hypothesis ("a blank cell was dropped at position k") and keeping
# only those where a single k makes EVERY column sum to the printed provincial
# Total exactly. Five rows in the corpus satisfy that test, each at exactly one k.
#
# 2026 w26 Mansehra was confirmed visually against the source PDF: the row carries
# nine numbers against ten column headings, the cell under "Animal / Dog Bite" is
# empty, and the value 217 sits under "TB" — as the Total-row arithmetic predicts,
# and NOT under "ALRI < 5 years" where the uncorrected transcription placed it.
#
# A candidate is accepted only when the week had TWO OR MORE columns failing to
# reconcile and the single-blank hypothesis repairs all of them at once. A week
# with just one column off by a small amount is NOT accepted: many transformations
# can absorb a lone off-by-one, so such a case cannot be distinguished from a
# rounding or arithmetic slip in the bulletin itself. On that rule 2024 w50
# (Abbottabad, one column off by 1) is deliberately left uncorrected.
#
# The blank position is given by COLUMN HEADING, not by index, and resolved
# against each week's own header at build time — the column order rotates week to
# week, so a hard-coded index would silently point at the wrong condition.
#
# (year, week, district_as_printed) -> heading the blank cell falls under
DROPPED_CELL_FIX = {
    (2026, 26, "Mansehra"): "Animal / Dog Bite",  # confirmed against the source PDF
    (2023, 16, "Malakand"): "Typhoid",
    (2023, 22, "Malakand"): "Dog Bite",
    (2023, 26, "Malakand"): "Dog Bite",
}

def apply_dropped_cell_fix(year, week, dist_raw, vals, headers):
    heading = DROPPED_CELL_FIX.get((year, week, dist_raw.strip()))
    if heading is None:
        return vals, None
    if heading not in headers:
        raise ValueError(f"{year} w{week}: correction names column {heading!r}, "
                         f"absent from that week's header {headers}")
    k = headers.index(heading)
    return list(vals[:k]) + [""] + list(vals[k:len(headers) - 1]), heading

def is_data_row(raw):
    d = raw.strip()
    if d == "":
        return False
    if d.lower().startswith("note:"):
        return False
    if d.lower() == "total":
        return False
    return True

# ---- parse ----
blocks = []
cur = None
for line in open(RAW, encoding="utf-8"):
    line = line.rstrip("\n")
    if line.startswith("### "):
        if cur: blocks.append(cur)
        y, w = line[4:].split()
        cur = {"year": int(y), "week": int(w), "rows": []}
    elif line.strip() == "":
        continue
    else:
        cur["rows"].append(line)
if cur: blocks.append(cur)

# The raw transcription file is in the order the bulletins were processed: the
# 8-bulletin feasibility sample first, then the remaining 156. Sort into
# chronological order so that every emitted file is time-ordered — otherwise any
# user who plots a series without sorting first gets a scrambled x-axis.
blocks.sort(key=lambda b: (b["year"], b["week"]))

long_rows = []          # year, week, district_raw, district_canonical, unit_type, parent_district, condition_raw, condition, cases
col_orders = []         # year, week, ordered canonical cols
presence = defaultdict(dict)
swat = {}
all_conditions = OrderedDict()
unmapped_districts = set()
skipped_rows = []       # (year, week, raw_line) for Note:/blank rows dropped
dropped_cell_rows = []  # (year, week, district, heading) rows repaired by DROPPED_CELL_FIX

def num(x):
    x = x.strip().replace(",", "")
    if x.upper() == "NR" or x == "":
        return None
    return int(x)

for b in blocks:
    header = [c.strip() for c in b["rows"][0].split("|")]
    cond_headers = header[1:]                # drop 'Districts'/'Diseases'
    canon_cols = [canon(h) for h in cond_headers]
    col_orders.append((b["year"], b["week"], canon_cols, cond_headers))
    for cc in canon_cols:
        all_conditions[cc] = True
        presence[(b["year"], b["week"])][cc] = True
    for r in b["rows"][1:]:
        cells = [c.strip() for c in r.split("|")]
        dist_raw = cells[0]
        if dist_raw.strip().lower() == "total":
            b["totals"] = [num(v) for v in cells[1:]]
            continue
        if not is_data_row(dist_raw):
            skipped_rows.append((b["year"], b["week"], r))
            continue
        cells_v = cells[1:]
        cells_v, fixed_heading = apply_dropped_cell_fix(b["year"], b["week"], dist_raw, cells_v, cond_headers)
        if fixed_heading:
            dropped_cell_rows.append((b["year"], b["week"], dist_raw, fixed_heading))
        dist_canon, unit_type, parent = canon_district(dist_raw, b["year"], b["week"])
        if unit_type == "UNMAPPED":
            unmapped_districts.add(dist_raw)
        for cond_raw, cc, val in zip(cond_headers, canon_cols, cells_v):
            long_rows.append([b["year"], b["week"], dist_raw, dist_canon, unit_type, parent or "",
                               cond_raw, cc, num(val)])
        if dist_canon == "Swat":
            swat[(b["year"], b["week"])] = {cc: num(v) for cc, v in zip(canon_cols, cells_v)}

# ---- data-quality flag: 2025 w9 table is byte-identical to 2025 w8 in the
# source PDF (confirmed by direct re-fetch of both bulletins) -- almost
# certainly NIH republished w8's table under the w9 bulletin by mistake.
# Flagged, not dropped: downstream users should exclude/impute w9 for any
# time-series or week-over-week analysis.
DUPLICATE_SOURCE_WEEKS = [((2025, 9), (2025, 8), "Table 4 in the w9 bulletin is byte-identical to w8's; likely a source-side republish error, not an extraction error. Confirmed by direct re-fetch of both PDFs on 2026-08-06.")]

# ---- write long panel ----
with open(f"{OUT}/idsr_kp_long.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["year", "epi_week", "district_as_printed", "district", "unit_type", "parent_district",
                "condition_as_printed", "condition", "cases"])
    w.writerows(long_rows)

# ---- column order per week (rotation evidence) ----
with open(f"{OUT}/table4_columns_by_week.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["year", "epi_week", "position", "condition", "as_printed"])
    for y, wk, canon_cols, raw_cols in col_orders:
        for i, (cc, rc) in enumerate(zip(canon_cols, raw_cols), 1):
            w.writerow([y, wk, i, cc, rc])

# ---- presence matrix condition x week ----
weeks = [(y, wk) for (y, wk, _, _) in col_orders]
conds = list(all_conditions.keys())
with open(f"{OUT}/condition_presence_matrix.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["condition"] + [f"{y}w{wk}" for y, wk in weeks])
    for c in conds:
        w.writerow([c] + ["Y" if presence[wk].get(c) else "" for wk in weeks])

# ---- Swat wide series ----
with open(f"{OUT}/idsr_kp_swat.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["year", "epi_week"] + conds)
    for y, wk in weeks:
        row = swat.get((y, wk), {})
        w.writerow([y, wk] + [row.get(c, "") for c in conds])

# ---- district crosswalk reference table ----
with open(f"{OUT}/district_crosswalk.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["raw_label", "canonical_unit", "unit_type", "parent_district",
                "applies_from", "applies_to", "note"])
    rows = [(raw, can, ut, par or "", "", "", "")
            for raw, (can, ut, par) in DISTRICT_XWALK.items()]
    # the two time-scoped SWA entries
    rows.append(("SWA", "South Waziristan (undivided)", "district", "",
                 "2023w34", "2024w49",
                 "Legacy 'South Waziristan Agency' label; denotes the UNDIVIDED "
                 "district while it is the only South Waziristan row reported. "
                 "Do not equate with either Lower or Upper."))
    rows.append(("SWA", "South Waziristan Upper", "district", "",
                 "2024w50", "2024w52",
                 "From 2024 w50 'South Waziristan (Lower)' reports separately and "
                 "SWA becomes the residual (Upper); continues as 'SWU' from 2025 w1."))
    for r in sorted(rows, key=lambda x: (x[0], x[4])):
        w.writerow(r)

# ---- data quality flags file ----
with open(f"{OUT}/data_quality_flags.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["flag_type", "year", "epi_week", "detail"])
    for (y, wk), (dy, dw), note in DUPLICATE_SOURCE_WEEKS:
        w.writerow(["duplicate_source_table", y, wk, f"identical to {dy} w{dw}: {note}"])
    for y, wk, raw_line in skipped_rows:
        w.writerow(["dropped_non_data_row", y, wk, raw_line])
    for d in sorted(unmapped_districts):
        w.writerow(["unmapped_district_label", "", "", d])
    for y, wk, d, h in dropped_cell_rows:
        w.writerow(["dropped_cell_corrected", y, wk,
                    f"{d}: the source row carries one fewer value than there are column headings. "
                    f"The blank cell falls under '{h}'; values after it were transcribed one column "
                    f"to the left and have been realigned. Corrected row reconciles exactly against "
                    f"the printed provincial Total on every column."])
    for (y, wk) in SWA_AMBIGUOUS_WEEKS:
        w.writerow(["overlapping_reporting_units", y, wk,
                    "'SWA' reports at undivided-district level (all-condition total 340) "
                    "while a small 'South Waziristan (Lower)' row (total 42) also appears. "
                    "The two rows overlap geographically in this week. SWA is coded as "
                    "South Waziristan (undivided); treat this week as ambiguous for any "
                    "South Waziristan analysis."])
    w.writerow(["reporting_unit_redefinition", SWA_CUTOVER[0], SWA_CUTOVER[1],
                "'SWA' changes meaning at this week: undivided South Waziristan before, "
                "South Waziristan Upper from here on (continues as 'SWU' from 2025 w1). "
                "See district_crosswalk.csv applies_from/applies_to."])

# ---- console report ----
print("Bulletins parsed:", len(blocks))
print("Distinct conditions ever seen (%d): %s" % (len(conds), ", ".join(conds)))
print()
# order-stability check
orders = set(tuple(cc) for _, _, cc, _ in col_orders)
print("Distinct column orderings across %d bulletins: %d  -> %s" %
      (len(col_orders), len(orders), "STABLE" if len(orders) == 1 else "NOT STABLE"))
print()
# totals descending check
nondesc = 0
for b in blocks:
    t = b.get("totals")
    if t:
        clean = [x for x in t if x is not None]
        mono = all(clean[i] >= clean[i+1] for i in range(len(clean)-1))
        if not mono:
            nondesc += 1
            print(f"  NOT descending: {b['year']} w{b['week']:>2}  {t}")
print("Weeks where Total row is NOT strictly descending:", nondesc, "/", len(blocks))
print()
# district count per week
ndistricts = [sum(1 for r in b["rows"][1:] if is_data_row(r.split('|')[0])) for b in blocks]
print("District rows per bulletin: min %d, max %d" % (min(ndistricts), max(ndistricts)))
print()
print("Unmapped district labels (need crosswalk entries):", sorted(unmapped_districts) or "none")
print("Non-data rows dropped (Note:/blank):", len(skipped_rows))
print()
# which core conditions present every week
always = [c for c in conds if all(presence[wk].get(c) for wk in weeks)]
sometimes = [c for c in conds if c not in always]
print("Present in EVERY bulletin (%d):" % len(always), ", ".join(always))
print("Rotating (present only some weeks) (%d):" % len(sometimes), ", ".join(sometimes))
print()
print("Data-quality flags written to data_quality_flags.csv (source duplicate weeks, dropped rows, unmapped labels)")
print()
# ---- integrity check: no two source rows may collapse onto one canonical key ----
seen = {}
collisions = []
for r in long_rows:
    key = (r[0], r[1], r[3], r[7])          # year, week, canonical district, condition
    if key in seen and seen[key] != r[2]:
        collisions.append((key, seen[key], r[2]))
    seen[key] = r[2]
print("Duplicate canonical keys (year, week, district, condition):", len(collisions))
if collisions:
    for k, a, bb in collisions[:10]:
        print("   COLLISION", k, "<-", a, "and", bb)
    raise SystemExit("FAILED: canonical district mapping is not injective within a week")
print("Canonical units:", len(set(r[3] for r in long_rows)))
print("Total long rows:", len(long_rows))
