#!/usr/bin/env python3
"""Generate an HTML memo comparing paper BibTeX against Crossref/web."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import bibtexparser
import requests
from bs4 import BeautifulSoup
from habanero import Crossref


ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT / "main.tex"
BIB_FILE = ROOT / "reports" / "pss_references.bib"
OUT_HTML = ROOT / "reports" / "bibtex_scrub_memo.html"
CACHE_FILE = ROOT / "reports" / "bibtex_scrub_cache.json"


def strip_latex_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        escaped = False
        kept = []
        for ch in line:
            if ch == "%" and not escaped:
                break
            kept.append(ch)
            escaped = ch == "\\" and not escaped
            if ch != "\\":
                escaped = False
        out.append("".join(kept))
    return "\n".join(out)


def cited_keys(tex: str) -> list[str]:
    tex = strip_latex_comments(tex)
    keys: list[str] = []
    seen = set()
    for match in re.finditer(r"\\(?:cite|citep|citet|textcite)(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        for key in match.group(1).split(","):
            key = key.strip()
            if key and not key.startswith("#") and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def load_bib(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    db = bibtexparser.loads(text)
    entries = {entry["ID"]: entry for entry in db.entries}
    raw: dict[str, str] = {}
    for match in re.finditer(r"(?ms)^@\w+\s*\{\s*([^,\s]+)\s*,.*?(?=^@\w+\s*\{|\Z)", text):
        raw[match.group(1)] = match.group(0).strip()
    return entries, raw


def norm(text: str | None) -> str:
    text = text or ""
    text = html.unescape(text)
    text = re.sub(r"<\?xmltex.*?\?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = text.replace("--", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def norm_doi(doi: str | None) -> str:
    doi = (doi or "").strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
    return doi.lower()


def year_from_crossref(item: dict[str, Any]) -> str:
    for field in ("published-print", "published-online", "issued"):
        parts = item.get(field, {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0]:
            return str(parts[0][0])
    return ""


def title_from_crossref(item: dict[str, Any]) -> str:
    return " ".join(item.get("title", []) or [])


def venue_from_crossref(item: dict[str, Any]) -> str:
    return " ".join(item.get("container-title", []) or [])


def title_match(a: str | None, b: str | None) -> bool:
    na = norm(a)
    nb = norm(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def token_overlap(a: str | None, b: str | None) -> float:
    stop = {"a", "an", "and", "as", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    aa = {tok for tok in norm(a).split() if len(tok) > 2 and tok not in stop}
    bb = {tok for tok in norm(b).split() if len(tok) > 2 and tok not in stop}
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / min(len(aa), len(bb))


def year_match(a: str | None, b: str | None) -> bool:
    return bool(a and b and str(a)[:4] == str(b)[:4])


def cache_get(cache: dict[str, Any], namespace: str, key: str) -> Any | None:
    return cache.get(namespace, {}).get(key)


def cache_put(cache: dict[str, Any], namespace: str, key: str, value: Any) -> None:
    cache.setdefault(namespace, {})[key] = value


def crossref_lookup(entry: dict[str, Any], cr: Crossref, cache: dict[str, Any]) -> dict[str, Any]:
    doi = norm_doi(entry.get("doi"))
    title = entry.get("title", "")
    cache_key = doi or norm(title)
    cached = cache_get(cache, "crossref", cache_key)
    if cached is not None:
        return cached
    result: dict[str, Any] = {"found": False, "error": None}
    try:
        if doi:
            item = cr.works(ids=doi).get("message", {})
        else:
            items = cr.works(query_title=title, limit=5).get("message", {}).get("items", [])
            item = max(items, key=lambda row: float(row.get("score", 0.0))) if items else {}
        if item:
            result = {
                "found": True,
                "title": title_from_crossref(item),
                "year": year_from_crossref(item),
                "venue": venue_from_crossref(item),
                "doi": item.get("DOI", ""),
                "type": item.get("type", ""),
                "url": item.get("URL", ""),
            }
    except Exception as exc:
        result = {"found": False, "error": str(exc)}
    cache_put(cache, "crossref", cache_key, result)
    return result


def ads_bibcode(entry: dict[str, Any]) -> str:
    adsurl = entry.get("adsurl", "")
    match = re.search(r"/abs/([^/?#]+)", adsurl)
    return match.group(1) if match else ""


def ads_year(doc: dict[str, Any]) -> str:
    year = doc.get("year")
    if year:
        return str(year)
    pubdate = doc.get("pubdate", "")
    return str(pubdate)[:4] if pubdate else ""


def ads_lookup(entry: dict[str, Any], api_key: str | None, cache: dict[str, Any]) -> dict[str, Any]:
    if not api_key:
        return {"checked": False, "error": "NASA_ADS_API_KEY is not set"}
    doi = norm_doi(entry.get("doi"))
    bibcode = ads_bibcode(entry)
    title = entry.get("title", "")
    cache_key = doi or bibcode or norm(title)
    cached = cache_get(cache, "ads", cache_key)
    if cached is not None:
        return cached

    queries = []
    if doi:
        queries.append(f'doi:"{doi}"')
    if bibcode:
        queries.append(f'bibcode:"{bibcode}"')
    if title:
        queries.append(f'title:"{title}"')

    result: dict[str, Any] = {"checked": True, "found": False, "error": None}
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "cape-cod-bay-bibtex-scrubber/1.0"}
    fields = "title,year,pubdate,bibcode,doi,pub,author,identifier"
    try:
        for query in queries:
            resp = requests.get(
                "https://api.adsabs.harvard.edu/v1/search/query",
                params={"q": query, "fl": fields, "rows": 5},
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
            if not docs:
                continue
            doc = max(docs, key=lambda row: token_overlap(" ".join(row.get("title", []) or []), title))
            result = {
                "checked": True,
                "found": True,
                "title": " ".join(doc.get("title", []) or []),
                "year": ads_year(doc),
                "venue": doc.get("pub", ""),
                "doi": "; ".join(doc.get("doi", []) or []),
                "bibcode": doc.get("bibcode", ""),
                "url": f"https://ui.adsabs.harvard.edu/abs/{doc.get('bibcode', '')}" if doc.get("bibcode") else "",
                "query": query,
                "error": None,
            }
            break
    except Exception as exc:
        result = {"checked": True, "found": False, "error": str(exc)}
    cache_put(cache, "ads", cache_key, result)
    return result


def web_lookup(entry: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    url = entry.get("url") or (f"https://doi.org/{entry['doi']}" if entry.get("doi") else "")
    if not url:
        return {"checked": False}
    cached = cache_get(cache, "web", url)
    if cached is not None:
        return cached
    result: dict[str, Any] = {"checked": True, "ok": False, "url": url, "error": None}
    try:
        resp = requests.get(url, timeout=20, allow_redirects=True, headers={"User-Agent": "cape-cod-bay-bibtex-scrubber/1.0"})
        body = resp.text[:300000]
        soup = BeautifulSoup(body, "html.parser")
        page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = soup.get_text(" ", strip=True)[:20000]
        expected = entry.get("title", "")
        final_url = resp.url
        text_ok = title_match(page_title, expected) or title_match(text, expected)
        doi_ok = norm_doi(entry.get("doi")) and norm_doi(entry.get("doi")) in norm_doi(body)
        github_ok = "github.com" in url and final_url.lower().startswith(url.lower().rstrip("/"))
        zenodo_ok = "zenodo" in url.lower() and (doi_ok or text_ok)
        same_site_ok = urlparse(final_url).netloc.lower() == urlparse(url).netloc.lower()
        title_overlap = token_overlap(page_title, expected)
        text_overlap = token_overlap(text, expected)
        generic_ok = resp.ok and (
            text_ok
            or doi_ok
            or github_ok
            or zenodo_ok
            or title_overlap >= 0.5
            or text_overlap >= 0.5
            or same_site_ok
        )
        result = {
            "checked": True,
            "ok": bool(generic_ok),
            "status": resp.status_code,
            "url": url,
            "final_url": final_url,
            "page_title": page_title,
            "text_match": bool(text_ok),
            "title_overlap": round(title_overlap, 3),
            "text_overlap": round(text_overlap, 3),
            "same_site": bool(same_site_ok),
            "doi_match": bool(doi_ok),
            "error": None,
        }
    except Exception as exc:
        result = {"checked": True, "ok": False, "url": url, "error": str(exc)}
    cache_put(cache, "web", url, result)
    return result


def is_url_only_entry(entry: dict[str, Any]) -> bool:
    url = (entry.get("url") or "").lower()
    if "github.com" in url or "zenodo" in url:
        return True
    return entry.get("ENTRYTYPE", "").lower() in {"misc", "online"}


def entry_needs_crossref(entry: dict[str, Any]) -> bool:
    if is_url_only_entry(entry):
        return False
    return entry.get("ENTRYTYPE", "").lower() in {"article", "book", "inbook", "incollection", "inproceedings"}


def status_badge(ok: bool, label: str) -> str:
    cls = "ok" if ok else "bad"
    symbol = "✓" if ok else "✕"
    return f'<span class="badge {cls}"><span class="box">{symbol}</span> {html.escape(label)}</span>'


def source_validates(entry: dict[str, Any], source: dict[str, Any]) -> bool:
    if not source.get("found"):
        return False
    checks = [title_match(source.get("title"), entry.get("title"))]
    if entry.get("year") and source.get("year"):
        checks.append(year_match(source.get("year"), entry.get("year")))
    if entry.get("doi") and source.get("doi"):
        source_dois = [norm_doi(part) for part in str(source.get("doi", "")).split(";")]
        checks.append(norm_doi(entry.get("doi")) in source_dois)
    return all(checks)


def result_status(entry: dict[str, Any], crossref: dict[str, Any], ads: dict[str, Any], web: dict[str, Any]) -> tuple[bool, list[str]]:
    checks: list[tuple[bool, str]] = []
    outcome_checks: list[bool] = []
    if entry_needs_crossref(entry):
        crossref_ok = source_validates(entry, crossref)
        checks.append((crossref_ok, "Crossref validates"))
        if ads.get("checked"):
            ads_ok = source_validates(entry, ads)
            checks.append((ads_ok, "ADS validates"))
            bibliographic_ok = crossref_ok or ads_ok
        else:
            bibliographic_ok = crossref_ok
        checks.append((bibliographic_ok, "Bibliographic record"))
        outcome_checks.append(bibliographic_ok)
    if web.get("checked"):
        web_ok = bool(web.get("ok"))
        checks.append((web_ok, "Web page"))
        outcome_checks.append(web_ok)
    if not checks:
        checks.append((False, "No external check"))
        outcome_checks.append(False)
    overall = all(outcome_checks) if outcome_checks else False
    badges = [status_badge(ok, label) for ok, label in checks]
    return overall, badges


def render_html(rows: list[dict[str, Any]]) -> str:
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #182026; background: #f7f8fa; }
h1 { margin-bottom: 0.2rem; }
.summary { color: #53606b; margin-bottom: 24px; }
.entry { background: white; border: 1px solid #d9dee3; border-radius: 8px; margin: 16px 0; overflow: hidden; }
.head { display: flex; gap: 12px; justify-content: space-between; align-items: center; padding: 14px 16px; border-bottom: 1px solid #e6e9ed; }
.key { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 700; }
.overall.ok { color: #0b7a3b; }
.overall.bad { color: #b42318; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; padding: 12px 16px; }
.badge { border-radius: 999px; padding: 3px 9px; font-size: 12px; border: 1px solid transparent; }
.badge.ok { background: #e9f8ef; color: #0b7a3b; border-color: #b7e5c6; }
.badge.bad { background: #fdecec; color: #b42318; border-color: #f3b9b4; }
.box { font-weight: 800; margin-right: 3px; }
.cols { display: grid; grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr); gap: 0; border-top: 1px solid #edf0f3; }
.panel { padding: 14px 16px; border-right: 1px solid #edf0f3; }
.panel:last-child { border-right: 0; }
.panel h3 { margin: 0 0 8px; font-size: 14px; }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; background: #101820; color: #e8edf2; padding: 12px; border-radius: 6px; font-size: 12px; }
dl { margin: 0; display: grid; grid-template-columns: 80px 1fr; gap: 5px 10px; font-size: 13px; }
dt { color: #66727d; }
dd { margin: 0; overflow-wrap: anywhere; }
"""
    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>BibTeX Scrub Memo</title>",
        f"<style>{css}</style></head><body>",
        "<h1>BibTeX Scrub Memo</h1>",
        "<p class='summary'>Compares cited BibTeX records in <code>reports/pss_references.bib</code> against Crossref via <code>habanero.Crossref</code> and NASA ADS via its API. Zenodo, GitHub, and other URL-style citations are checked by loading the cited webpage.</p>",
    ]
    for row in rows:
        overall_cls = "ok" if row["overall"] else "bad"
        overall_label = "OK" if row["overall"] else "CHECK"
        out.append("<section class='entry'>")
        out.append(f"<div class='head'><div><span class='key'>{html.escape(row['key'])}</span> <span>{html.escape(row['type'])}</span></div><strong class='overall {overall_cls}'>{overall_label}</strong></div>")
        out.append("<div class='badges'>" + "".join(row["badges"]) + "</div>")
        out.append("<div class='cols'>")
        out.append("<div class='panel'><h3>BibTeX Entry In Paper</h3><pre>" + html.escape(row["raw"]) + "</pre></div>")
        out.append("<div class='panel'><h3>External Results</h3>")
        out.append("<h3>Crossref</h3>" + render_dl(row["crossref"]))
        out.append("<h3>NASA ADS</h3>" + render_dl(row["ads"]))
        out.append("<h3>Web Page</h3>" + render_dl(row["web"]))
        out.append("</div></div></section>")
    out.append("</body></html>")
    return "\n".join(out)


def render_dl(data: dict[str, Any]) -> str:
    if not data:
        return "<p>Not checked.</p>"
    keys = [
        "found",
        "checked",
        "ok",
        "title",
        "year",
        "venue",
        "doi",
        "bibcode",
        "url",
        "final_url",
        "page_title",
        "query",
        "status",
        "title_overlap",
        "text_overlap",
        "same_site",
        "error",
    ]
    rows = []
    for key in keys:
        if key in data and data[key] not in (None, ""):
            rows.append(f"<dt>{html.escape(key)}</dt><dd>{html.escape(str(data[key]))}</dd>")
    return "<dl>" + "".join(rows) + "</dl>" if rows else "<p>Not checked.</p>"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main", type=Path, default=MAIN_TEX)
    p.add_argument("--bib", type=Path, default=BIB_FILE)
    p.add_argument("--out", type=Path, default=OUT_HTML)
    p.add_argument("--cache", type=Path, default=CACHE_FILE)
    p.add_argument("--ads-api-key", default=os.environ.get("NASA_ADS_API_KEY"), help="NASA ADS API key. Defaults to NASA_ADS_API_KEY.")
    p.add_argument("--keys", nargs="*", help="Optional subset of citation keys.")
    args = p.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}
    entries, raw_entries = load_bib(args.bib)
    keys = args.keys or cited_keys(args.main.read_text(encoding="utf-8"))
    cr = Crossref()
    rows = []
    missing = []

    total = len(keys)
    for idx, key in enumerate(keys, start=1):
        entry = entries.get(key)
        if not entry:
            missing.append(key)
            continue
        print(f"[{idx}/{total}] {key}", flush=True)
        crossref = crossref_lookup(entry, cr, cache) if entry_needs_crossref(entry) else {}
        ads = ads_lookup(entry, args.ads_api_key, cache) if entry_needs_crossref(entry) else {}
        web = web_lookup(entry, cache) if entry.get("url") and not entry_needs_crossref(entry) else {"checked": False}
        overall, badges = result_status(entry, crossref, ads, web)
        rows.append(
            {
                "key": key,
                "type": entry.get("ENTRYTYPE", ""),
                "raw": raw_entries.get(key, ""),
                "crossref": crossref,
                "ads": ads,
                "web": web,
                "overall": overall,
                "badges": badges,
            }
        )
        args.cache.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    args.cache.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    args.out.write_text(render_html(rows), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"entries: {len(rows)}")
    if missing:
        print(f"missing BibTeX keys: {', '.join(missing)}")


if __name__ == "__main__":
    main()
