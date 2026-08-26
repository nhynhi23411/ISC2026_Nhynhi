# District-week notifiable infectious disease surveillance data for Khyber Pakhtunkhwa, Pakistan, 2023–2026

This dataset is the structured panel described in the Data in Brief article of the same title. It
is derived from Table 4 — the Khyber Pakhtunkhwa province district-wise surveillance table — of 164
weekly Integrated Disease Surveillance and Response (IDSR) Public Health Bulletins published by the
National Institute of Health, Pakistan.

**Coverage:** 2023 week 16 and weeks 22–52, all of 2024, all of 2025, and 2026 weeks 1–28 — 164 of
the 169 weeks in that window. The five absent weeks (2023 weeks 17–21) are missing from the
publisher's own archive, not from retrieval, and are listed in `bulletin_links_verified.csv`.

**Scale:** 59,824 district-week-condition observations, 44 canonical reporting units, 16 notifiable
conditions.

---

## Files

| File | Contents |
|---|---|
| `idsr_kp_long.csv` | **Primary file.** The full panel in long (tidy) format, one row per district per week per condition. |
| `table4_columns_by_week.csv` | The exact column position and canonical mapping of every condition in each of the 164 bulletins. |
| `condition_presence_matrix.csv` | A 16 × 164 condition-by-week grid of which conditions appeared each week. |
| `district_crosswalk.csv` | Every raw district/sub-division label mapped to a canonical unit, with validity windows for labels whose meaning changes. |
| `data_quality_flags.csv` | Machine-readable flags for every known issue (see below). |
| `idsr_kp_swat.csv` | Swat district's row for every week, in wide format. Convenience extract. |
| `swat_acute_diarrhoea_series.csv` | Swat's acute diarrhoea (non-cholera) time series. Convenience extract. |
| `bulletin_links_verified.csv` | The verified retrieval URL for every bulletin used, plus the five confirmed-absent weeks. |
| `tables_raw.txt` | The raw transcribed source text for all 164 bulletins, in `### YEAR WEEK` blocks. |
| `build.py` | The parser that produces every CSV above from `tables_raw.txt`. |

Every file is sorted chronologically. `tables_raw.txt` and `build.py` are included so that the path
from transcribed source to published panel is fully reproducible: running `build.py` over
`tables_raw.txt` regenerates the CSV files exactly.

## Fields in `idsr_kp_long.csv`

| Field | Meaning |
|---|---|
| `year`, `epi_week` | Epidemiological year and week of the bulletin. |
| `district_as_printed` | The district or sub-division label exactly as printed in the source. |
| `district` | Canonical reporting unit (see `district_crosswalk.csv`). |
| `unit_type` | `district`, `subdivision`, or `typo`. |
| `parent_district` | For sub-division rows, the district they sit within. Blank otherwise. |
| `condition_as_printed` | The column heading exactly as printed in the source. |
| `condition` | Canonical condition name (16 values). |
| `cases` | Case count. **Blank means "NR" (not reported) in the source, which is not the same as a reported zero.** |

---

## Three things to know before using this data

**1. Absence of a condition is not a zero.** From 2023 week 32 onward the source table reports the
ten conditions with the highest provincial case totals that week, re-ranked weekly, producing 106
distinct column orderings across the corpus. Six conditions appear in all 164 bulletins — acute
diarrhoea (non-cholera), malaria, influenza-like illness, ALRI in children under five, bloody
diarrhoea and typhoid — and are complete. The other ten rotate in and out. When one of those ten is
absent from a week it means only that it did not rank that week: this is right-censoring, and
treating it as a zero will bias any analysis. The 11 weeks before the transition (2023 week 16 and
weeks 22–31) used a fixed layout and are not censored.

**Parse by column header, never by column position.**

**2. One district label changes meaning mid-series.** "SWA" denotes the *undivided* South Waziristan
district through 2024 week 49, and its *Upper* portion from 2024 week 50, continuing as "SWU" from
2025 week 1. `district_crosswalk.csv` carries `applies_from` / `applies_to` for the two time-scoped
entries; every other label maps constantly. A static label-to-district mapping conflates the
undivided district with one of its successors and corrupts both South Waziristan series. There is no
separate Lower/Upper series before 2024 week 50.

**3. No denominators.** The bulletins report no reporting-facility counts and no catchment
populations. All values are raw case counts, not incidence rates, and reporting completeness is not
observable from the data. Apparent between-district differences may reflect reporting effort as much
as disease burden.

---

## Known issues, all flagged in `data_quality_flags.csv`

- **`duplicate_source_table`** — Table 4 of the 2025 week 9 bulletin reproduces week 8's exactly.
  Confirmed by re-retrieving both PDFs. Retained for transparency; exclude or impute it in any
  week-over-week analysis.
- **`dropped_cell_corrected`** — four rows (Mansehra 2026 w26; Malakand 2023 w16, w22, w26) in which
  the source row contains a blank cell and the transcription had left-aligned the remaining values,
  assigning them to the wrong conditions. Corrected. The 2026 w26 case was confirmed against the
  source PDF and moved 217 cases between ALRI<5 and TB.
- **`reporting_unit_redefinition`** and **`overlapping_reporting_units`** — the 2024 w50 "SWA"
  redefinition, and 2024 w49 where an undivided-level SWA row and a small Lower row appear together.
- **`dropped_non_data_row`** — four explanatory notes in the source (2024 weeks 36, 37, 40, 41)
  recording that Kohat and/or Battagram were not reported that week. Logged, not treated as data.

**Residual discrepancies.** After correction, 137 of the 163 weeks carrying a printed Total row
reconcile exactly on every column. In the other 26 at least one column does not sum to the printed
provincial total; 19 differ by no more than 10 cases. The largest, 2023 week 41, was checked against
its PDF — the row count and the suspect district's values both match the source — so that
inconsistency is internal to the published bulletin, not introduced here.

---

## Provenance and limitations

Table 4 was transcribed from each bulletin PDF using an AI-assisted document-reading tool, with
headers, row labels and cell values preserved as printed. Structural findings were validated by
internal consistency checks across the corpus (descending-Total test, cell-count test, district-to-
total reconciliation) and by re-retrieving source PDFs. Individual cell values have not been
systematically verified against every source PDF by an independent method; the reconciliation check
constrains but does not eliminate residual error, since it detects only discrepancies that disturb a
column total. Users needing exact counts for a specific district-week should confirm them against
that bulletin, whose URL is in `bulletin_links_verified.csv`.

## Citation

Please cite the accompanying Data in Brief article. The source bulletins remain the work of the
National Institute of Health, Pakistan, and are publicly available at https://www.nih.org.pk/

## Licence

CC BY 4.0. The underlying bulletins are public documents of a national public health authority.
