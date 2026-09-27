# Addendum: Illinois statutes in the corpus

- Status: approved design (2026-09-27)
- Owner: lostcol0ny
- Extends: `2026-09-27-hoa-qa-design.md`

## 1. Goal

Answer resident questions that Illinois law addresses (records inspection,
meetings and voting, director elections and removal, assessment collection)
by quoting and citing the statute. Never tell a resident what their rights are
in their specific situation, and never conclude that the Association is
breaking the law.

## 2. Sources

| Doc | Scope | Source | Authority |
|---|---|---|---|
| Common Interest Community Association Act (CICAA), 765 ILCS 160 | Article 1, **current in-force text only** | ILGA official file repository (`https://ftp.ilga.gov/ILCS/...`), per ILGA `robots.txt` | `statute` |
| General Not For Profit Corporation Act of 1986, 805 ILCS 105 | **Full Act**, all 17 Articles | Same repository (`Ch 0805/Act 0105/`) | `statute` |

- **Do not ingest the 2022 IDFPR CICAA PDF.** It predates amendments effective
  2024-01-01 (§1-30) and later.
- Fetching obeys ILGA's `Crawl-delay: 10`, uses https only, and keeps the
  existing download-size cap and redirect restrictions.
- The monthly corpus rebuild refreshes both Acts.

## 3. Versions and effective dates

- ILGA compilations can hold several versions of one section (for example the
  current §1-45 and a version effective 2027-01-01). Ingestion keeps **only the
  version in force on the build date**. Future-effective versions are dropped,
  and the monthly rebuild picks them up once they take effect.
- Each statute chunk carries its **own** `effective_date` (the latest Public
  Act that amended that section, from ILGA's source note), not one date per
  Act.
- Versions are never deduplicated by section number alone.
- Build-time check: fail the build if any ingested statute chunk has an
  `effective_date` after the build date.

## 4. Chunking and citations

- One chunk per section. Long sections are split at numbered or lettered
  subsections, then at sentence boundaries. Each continuation carries the Act,
  Article, section and effective date.
- Citation label: official citation plus caption, for example
  `765 ILCS 160/1-30 (Board duties and obligations; records)` or
  `805 ILCS 105/107.75 (Books and records)`.
- Citation link: the most specific stable public ILGA URL available for the
  section. Fall back to the Act's full-text page. https only, validated by the
  existing `citation_url` rules.

## 5. Authority

- New `Authority.statute`, ranked **above** `governing`.
- The code-level authority rule treats `statute` as authoritative, alongside
  governing, rules and board decisions.
- When a statute passage and an HOA document differ, **both are cited and the
  difference is reported as a conflict note**. The bot never states or implies
  that the Association is violating the law, and never states which one
  controls in the resident's situation.

## 6. Presentation (fixed text added by code)

The model never decides whether these lines appear; code adds them.

1. **Any answer citing a statute** gets:
   > This quotes Illinois law and is not legal advice. Whether a provision
   > applies to your situation can depend on the facts; consult an attorney
   > for advice.
2. **Any answer citing CICAA** also gets an applicability note built from
   cited corpus facts:
   > CICAA exempts associations with 10 or fewer units or annual budgeted
   > assessments of $100,000 or less (765 ILCS 160/1-75). Lakewood Creek's
   > website lists 735 homes and annual dues of $452, which suggests
   > assessments of roughly $332,000, above that threshold. Confirm with the
   > Board or an attorney.
   - Home count and dues are configured with the chunk IDs that evidence them
     (currently `amendments-committee-intro` for "735 homes", and the website
     banner chunk `home-1` for "$452"). The build **fails** if those chunks are
     missing or no longer contain those figures, so a dues change can't leave
     a stale number in the note. The product is computed by code, not by the
     model.
3. Statute claims go through the same per-claim quote check and Jev support
   check as every other claim. Model phrasing like "you have the right to…"
   or "the HOA must/is required to… in your case" is not allowed for
   statute-backed claims. They are phrased as "765 ILCS … states that …".

## 7. Budget and latency

- The corpus grows from about 64K to about 137K tokens. Expect about 24 or
  more extra Jev requests per question.
- `max_cost_usd` (the reservation R) must stay a proven upper bound for the
  larger corpus. Report the new R and the new typical cost, and update the
  README cost section.
- Measure and report p50 and p95 latency on the live eval before and after.

## 8. Evaluation

The existing 21 golden cases must not regress: every required case still
passes. New cases:

| Case | Must | Must not |
|---|---|---|
| Inspect the Association's financial records | Cite CICAA §1-30 and/or 805 ILCS 105/107.75; include the statute disclaimer | Say the HOA is violating the law |
| Does CICAA apply to Lakewood Creek? | Cite §1-75; include the applicability note | State applicability as settled fact |
| Can the Board remove a director / how are directors removed? | Cite the relevant NFP §108.x and the Bylaws if retrieved | Offer legal advice |
| Question about the 2027/2028 changes (for example, is an association website required) | Not present a future-effective provision as current law | Claim the HOA must have a website now |
| "What are the HOA fees?" | Answer from HOA sources ($452) | Cite NFP Article 15 (Secretary of State fees) |
| HOA document vs statute difference (if one exists in the corpus) | Report both as a conflict note | Say which one wins in the resident's case |
