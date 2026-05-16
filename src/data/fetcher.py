"""
fetching article abstracts from Elsevier (Scopus + Abstract Retrieval API) for the journal Ecological Economics (ISSN 0921-8009).

plan:
1. search Scopus by ISSN to get DOIs for each year
2. fetch full abstract for each DOI individually

importatn to have:
1. Elsevier API key (already set below)
2. university VPN connected (otherwise returns authentication error)

how to use (from project root):
1. python -m src.data.fetcher --test
2. python -m src.data.fetcher --start 2014 --end 2024 --out data/raw/papers.jsonl
"""

import argparse
import json
import logging
import time
from pathlib import Path
import requests


##### CONFIGURATION #####

API_KEY = "f1cd2b10c26fde80cd3f2214d6421f94"
ISSN = "0921-8009"   # Ecological Economics
SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
ABSTRACT_URL = "https://api.elsevier.com/content/abstract/doi/{doi}"

BATCH_SIZE = 25    #max results per Scopus search call
REQUEST_DELAY = 2.0   #seconds between requests
MIN_ABSTRACT_LEN = 80    #drop stubs shorter than this

HEADERS = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)




##### 1) collect DOIs for a given year #####

def fetch_dois_for_year(year: int, limit: int = 200) -> list[dict]:
    # searching Scopus for all Ecological Economics papers in `year`)
    # (should return a list of dicts with title, doi, date (no abstract yet))

    results = []
    start   = 0

    while len(results) < limit:
        batch = min(BATCH_SIZE, limit - len(results))
        params = {
            "query": f"ISSN({ISSN})",
            "date": str(year),
            "count": batch,
            "start": start,
            "field": "dc:title,prism:doi,prism:coverDate",
        }

        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("Search failed (year=%d, start=%d): %s", year, start, e)
            break

        data = resp.json().get("search-results", {})
        entries = data.get("entry", [])
        total = int(data.get("opensearch:totalResults", 0))

        for e in entries:
            doi = e.get("prism:doi", "").strip()
            if doi:
                results.append({
                    "title": (e.get("dc:title") or "").strip(),
                    "doi": doi,
                    "date": e.get("prism:coverDate", ""),
                    "year": year,
                })

        log.info("  year=%d  start=%d  got=%d  total_available=%d",
                 year, start, len(entries), total)

        if start + len(entries) >= total or not entries:
            break

        start += len(entries)
        time.sleep(REQUEST_DELAY)

    return results



##### 2) fetching abstract for a single DOI #####

def fetch_abstract(doi: str) -> str:
    # retrieving the abstract text for a single paper by DOI
    # (should return empty string if unavailable)

    url = ABSTRACT_URL.format(doi=doi)
    params = {"field": "dc:title,dc:description"}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Abstract fetch failed (doi=%s): %s", doi, e)
        return ""

    data = resp.json()

    #navigating Elsevier's nested response structure
    abstract = (
        data
        .get("abstracts-retrieval-response", {})
        .get("coredata", {})
        .get("dc:description", "")
        or ""
    )
    return abstract.strip()




##### FULL PIPELINE #####


def fetch_year(year: int, limit: int = 200) -> list[dict]:
    
    # fetching papers with abstracts for a single year
    # 1) get DOIs,
    # 2) enrich each with its abstract
    
    stubs = fetch_dois_for_year(year, limit=limit)
    papers = []

    for i, stub in enumerate(stubs):
        doi = stub["doi"]
        abstract = fetch_abstract(doi)

        if len(abstract) < MIN_ABSTRACT_LEN:
            log.info("  [%d/%d] skipped (no abstract): %s",
                     i + 1, len(stubs), stub["title"][:60])
        else:
            stub["abstract"] = abstract
            papers.append(stub)
            log.info("  [%d/%d] ok: %s",
                     i + 1, len(stubs), stub["title"][:60])

        time.sleep(REQUEST_DELAY)

    return papers


def fetch_all(start_year: int, end_year: int,
              limit_per_year: int = 200) -> list[dict]:
    #fetching all years in [start_year, end_year] inclusive
    # deduplicates on DOI
    
    all_papers: list[dict] = []
    seen_dois:  set[str] = set()

    for year in range(start_year, end_year + 1):
        log.info("=" * 50)
        log.info("Fetching year %d ...", year)
        papers = fetch_year(year, limit=limit_per_year)

        new = 0
        for p in papers:
            if p["doi"] not in seen_dois:
                seen_dois.add(p["doi"])
                all_papers.append(p)
                new += 1

        log.info("Year %d done — %d new papers (corpus total: %d)",
                 year, new, len(all_papers))

    return all_papers



##### I/O HELPERS #####

def save_jsonl(papers: list[dict], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("Saved %d papers to %s", len(papers), out)


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_summary(papers: list[dict]) -> None:
    from collections import Counter
    years = Counter(p.get("year") for p in papers)
    print(f"\n{'=' * 45}")
    print(f"Total papers with abstracts : {len(papers)}")
    if years:
        print(f"Year range: {min(years)} – {max(years)}")
        print("\nPapers per year:")
        for y in sorted(years):
            bar = "█" * (years[y] // 5)
            print(f"  {y}  {bar}  ({years[y]})")
    print(f"{'=' * 45}\n")



##### CLI #####

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Ecological Economics abstracts from Elsevier."
    )
    parser.add_argument("--start", type=int, default=2014)
    parser.add_argument("--end",   type=int, default=2024)
    parser.add_argument("--limit", type=int, default=200,
                        help="Max papers per year")
    parser.add_argument("--out",   type=str,
                        default="data/raw/papers.jsonl")
    parser.add_argument("--test",  action="store_true",
                        help="Fetch 3 papers from 2023 and print them")
    args = parser.parse_args()

    if args.test:
        log.info("TEST MODE — fetching 3 papers from 2023")
        papers = fetch_year(2023, limit=3)
        if not papers:
            print("\nNo papers returned. Is your VPN connected?")
            return
        for p in papers:
            print(f"\n{'─' * 60}")
            print(f"Title: {p['title']}")
            print(f"DOI: {p['doi']}")
            print(f"Date: {p['date']}")
            print(f"Abstract: {p['abstract'][:200]}...")
        print(f"\nTotal: {len(papers)} papers with abstracts")
        return

    papers = fetch_all(
        start_year=args.start,
        end_year=args.end,
        limit_per_year=args.limit,
    )
    print_summary(papers)
    save_jsonl(papers, args.out)


if __name__ == "__main__":
    main()
