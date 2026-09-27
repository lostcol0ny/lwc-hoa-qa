The Bylaws and Rules fixtures are short, verbatim text-layer excerpts from the
public PDFs identified in sources.yaml, including original OCR errors. Sections
are truncated after their first few lines to keep the fixtures small.

`golden_pages.json` holds real extracted text for Declaration Article 8 (PDF
pp. 26–37) and Rules 2023 Section 3 (PDF pp. 8–11): each page's heading lines plus
one body line per section, with the page's resolved printed number.
`golden_sections.json` is the reviewed list of chunk ids and citation labels those
excerpts must produce. Regenerate both only after reviewing a real build.

The minutes and Draft.js HTML fixtures retain the December 29, 2022 agenda and
budget vote wording. Committed fixtures never contain an individual's name
(board members, staff, attorneys, contacts): names are omitted or replaced with
placeholders such as `[Board Member A]`; surrounding markup is synthetic. Other
tests use synthetic contacts and documents. No network or API credentials are
used by these tests.
