"""Extract page-aware text, retaining source structure and PDF link indices."""

import json
import re
from dataclasses import dataclass

import pymupdf
from bs4 import BeautifulSoup

from hoa_qa.ingest.sources import Source


@dataclass(frozen=True)
class Page:
    text: str
    number: int | None = None
    printed: str | None = None
    heading: str | None = None


def html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("script, style, nav, footer"):
        tag.decompose()
    for tag in soup.select("br"):
        tag.replace_with("\n")
    for tag in soup.select("p, div, li, h1, h2, h3, h4"):
        tag.insert_before("\n")
        tag.insert_after("\n")
    lines = [
        re.sub(r"[^\S\n]+", " ", line).strip() for line in soup.get_text().splitlines()
    ]
    return "\n".join(line for line in lines if line)


def blog_text(html: str) -> str:
    # raw_decode stops at the JSON object's end; braces/semicolons inside strings
    # and trailing assignments cannot truncate the body as a greedy regex would.
    match = re.search(r"(?:window\s*\.\s*)?_BLOG_DATA\s*=\s*", html)
    if not match:
        raise ValueError("missing _BLOG_DATA assignment")
    try:
        data, _ = json.JSONDecoder().raw_decode(html[match.end() :].lstrip())
        content = data["post"]["fullContent"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty post.fullContent")
        if content.lstrip().startswith("{"):
            body = json.loads(content)
            blocks = body["blocks"]
            if not isinstance(blocks, list) or any(
                not isinstance(block.get("text"), str) for block in blocks
            ):
                raise ValueError("invalid Draft.js blocks")
            text = "\n".join(block["text"] for block in blocks)
        else:
            text = html_text(content)
        if not text:
            raise ValueError("empty post body")
        return text
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid _BLOG_DATA.post.fullContent") from exc


def web_pages(html: str, doc_id: str) -> list[Page]:
    soup = BeautifulSoup(html, "html.parser")
    if doc_id == "home":
        banner = soup.select_one('[data-aid="BANNER_TEXT_RENDERED"]')
        if banner is None:
            raise ValueError("home assessment banner missing")
        return [Page(html_text(str(banner)), heading="Assessment banner")]
    questions = soup.select('[data-aid^="FAQ_QUESTION_RENDERED_"]')
    if questions:
        pages = []
        for question in questions:
            suffix = str(question.get("data-aid")).rsplit("_", 1)[1]
            answer = soup.select_one(f'[data-aid="FAQ_ANSWER_RENDERED_{suffix}"]')
            if answer is None:
                raise ValueError(f"FAQ answer {suffix} missing")
            heading = question.get_text(" ", strip=True)
            pages.append(Page(heading + "\n" + html_text(str(answer)), heading=heading))
        return pages
    about = soup.select('[data-aid^="ABOUT_DESCRIPTION_RENDERED"]')
    if about:
        return [Page(html_text(str(tag)), heading="HOA Overview") for tag in about]
    # Fail on site redesign instead of silently ingesting navigation/cookie chrome.
    raise ValueError(f"main-content selector missing for {doc_id}")


def extract(source: Source, data: bytes) -> list[Page]:
    if source.kind == "blog_post":
        return [Page(blog_text(data.decode("utf-8")))]
    if source.kind == "web_page":
        return web_pages(data.decode("utf-8"), source.doc_id)
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as document:
        if any(n > len(document) for n in source.exclude_pages):
            raise ValueError(f"{source.doc_id}: exclusion beyond PDF length")
        for number in range(1, len(document) + 1):
            page = document[number - 1]
            if number in source.exclude_pages:
                continue
            text = page.get_text("text", sort=True)
            if not isinstance(text, str):
                raise TypeError("PDF text extraction returned non-text")
            # Only isolated digits in the bottom margin are printed pagination.
            printed = None
            for word in page.get_text("words"):
                if (
                    word[1] > page.rect.height * 0.90
                    and page.rect.width * 0.4 < word[0] < page.rect.width * 0.6
                    and re.fullmatch(r"\d{1,3}", word[4])
                ):
                    printed = word[4]
            if printed:
                text = re.sub(r"\n\s*" + printed + r"\s*$", "", text)
            if not text.strip():
                raise ValueError(
                    f"{source.doc_id} PDF p.{number}: no text layer; review and "
                    "document an exclusion or OCR before building"
                )
            pages.append(Page(text, number, printed))
    return pages
