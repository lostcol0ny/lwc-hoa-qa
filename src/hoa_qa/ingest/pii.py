"""Fail closed on unexpected contact information; approved business contacts only."""

import re

PHONE = re.compile(
    r"(?<!\d)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})"
    r"[ .-]\s*\d{3}[ .-]\s*\d{4}(?!\d)"
)
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
ADDRESS = re.compile(
    r"\b\d{1,6}\s+(?!RIGHT\s|Correction\s)(?:(?:[NSEW]\.?|North|South|East|West)\s+)?"
    r"(?:[A-Za-z][\w'-]*\s+){1,4}"
    r"(?:Dr(?:ive)?|St(?:reet)?|Ave(?:nue)?|Ct|Court|Ln|Lane|Rd|Road|"
    r"Blvd|Boulevard|Way|Pl(?:ace)?|Pkwy|Parkway)\b",
    re.I,
)

# doc_id -> exact normalized match -> documented business purpose.
ALLOWLIST: dict[str, dict[str, str]] = {
    "*": {
        "6302290092": "HOA clubhouse office",
        "6302290254": "HOA clubhouse fax (rental form and newsletter)",
        "2799 oakmont dr": "HOA clubhouse postal address",
        "2799 oakmont drive": "HOA clubhouse postal address (expanded suffix)",
        "lakewoodcreek@comcast.net": "HOA public organizational mailbox",
    },
    "declaration": {
        "2500 w higgins road": "Original HOA principal office, PDF p.56",
        "222 north lasalle street": "Recording law firm return address, PDF p.57",
        "3122363003": "Recording law firm office telephone, PDF p.57",
    },
    "bylaws": {
        "2500 w higgins road": "Original HOA principal office, Article 1",
    },
    "articles": {
        "2500 west niggins road": "OCR of original HOA business office, PDF p.2",
        "2500 west higgina road": "OCR of original HOA business office, PDF p.2",
        "2500 west higgins road": "Original HOA business office, PDF p.2",
        "118 wesp edward street": "OCR of incorporator corporation office, PDF p.3",
        "118 west edward street": "Incorporator corporation office, PDF p.3",
    },
    "clubhouse": {
        "6302735547": "Published rental attendant emergency contact, PDF p.2",
    },
    "faq": {
        "8446333577": "LRS municipal refuse service customer support",
        "montgomery@lrsrecycles.com": "LRS municipal refuse service mailbox",
        "4694902805": "MuniCap published special-assessment business contact",
        "8666488482": "MuniCap main office",
        "6308968080": "Village Planner office (extension 9022)",
    },
}


def normalized_hit(hit: str, phone: bool = False) -> str:
    if phone:
        return re.sub(r"\D", "", hit).removeprefix("1")
    return (
        " ".join(hit.lower().replace(".", "").split())
        if "@" not in hit
        else hit.lower()
    )


def check_pii(text: str, doc_id: str) -> None:
    allowed = ALLOWLIST["*"] | ALLOWLIST.get(doc_id, {})
    hits = []
    for pattern in (PHONE, EMAIL, ADDRESS):
        for match in pattern.finditer(text):
            if normalized_hit(match.group(), pattern is PHONE) not in allowed:
                hits.append(match.group())
    if hits:
        raise ValueError(f"PII check failed for {doc_id}: unexpected contacts {hits!r}")
