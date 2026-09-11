"""
WordPress plugin seam miner (bounded runner).

Aggregates public WordPress.org data (plugin API, review pages, version stats)
into a market-research dataset. Full plugin pull is cheap; the review-scraping
enrichment is bounded to the top-ranked candidates so the run stays polite to
wordpress.org (rate-limited, capped page depth).

Usage:
    pip install requests beautifulsoup4
    python wp_seam_miner.py

Outputs (in ./results/):
    all_plugins.csv   every plugin pulled, with metrics
    candidates.csv    filtered candidates; top ENRICH_TOP enriched
    summary.md        ranked table for reading
"""

import csv, os, re, time, math
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

API = "https://api.wordpress.org/plugins/info/1.2/"
HDR = {"User-Agent": "seam-miner/1.0 (research; contact via wordpress.org profile)"}
SLEEP = 1.5
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# Optional Firecrawl (firecrawl.dev) routing for the HTML page scrapes
# (review pages + advanced page). When FIRECRAWL_API_KEY is set, those fetches
# go through Firecrawl's proxy so a larger run doesn't get this IP throttled;
# otherwise they fall back to direct requests. The JSON APIs (plugin info,
# version stats) never use Firecrawl -- they are plain APIs, not scraping.
# Budget note: Firecrawl bills ~1 credit per page; the free tier is ~500-1000
# credits/month. The default bounded run is ~ENRICH_TOP * REVIEW_PAGES fetches.
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
FIRECRAWL_URL = os.environ.get("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v2/scrape")
_fc_credits = {"pages": 0}

POPULAR_PAGES = 2
TAGS = ["forms", "booking", "appointments", "membership", "lms", "backup",
        "woocommerce", "multilingual", "table", "directory", "caching", "seo"]
TAG_PAGES = {"woocommerce": 3}

MIN_INSTALLS = 10_000
MIN_REVIEWS = 20
MAX_AVG = 4.0
MIN_ONE_STAR_PCT = 15.0

# Bounds for the expensive review-scraping enrichment.
ENRICH_TOP = 30        # only enrich the top-N candidates by pre-score
REVIEW_PAGES = 12      # cap review pages per candidate (recent-first)

CAPTIVE_AUTHORS = {
    "automattic", "woocommerce", "wordpressdotorg", "jetpack", "crowdsignal",
    "elemntor", "yoast", "wpmudev", "wpengine", "deliciousbrains", "stellarwp",
    "liquidweb", "godaddy", "kinsta", "hostinger", "siteground", "cloudways",
    "facebook", "google", "microsoft", "tiktok", "pinterest", "klaviyo",
    "cloudflare", "mailchimp", "brevo", "mailerlite", "stripe", "paypal",
    "smub", "wpbeginner", "awesomemotive", "optinmonster", "exactmetrics",
    "monsterinsights", "wpforms", "aioseo", "wpmedia", "10up", "melapress",
    "wpdeveloper", "yithemes", "themeisle", "brainstormforce", "wpengine",
}


def get(url, **kw):
    for attempt in range(4):
        r = requests.get(url, headers=HDR, timeout=30, **kw)
        if r.status_code == 429:
            time.sleep(15 * (attempt + 1)); continue
        return r
    return r


class _Resp:
    """Minimal response shim so Firecrawl output is a drop-in for requests.get."""
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def _encode(url, params):
    if not params:
        return url
    from urllib.parse import urlencode
    return f"{url}?{urlencode(params)}"


def fetch_html(url, params=None):
    """Fetch an HTML page. Routes through Firecrawl when FIRECRAWL_API_KEY is
    set (with a direct-request fallback on any error); otherwise fetches
    directly. Returns an object exposing .status_code and .text."""
    full = _encode(url, params)
    if FIRECRAWL_KEY:
        try:
            r = requests.post(
                FIRECRAWL_URL,
                headers={"Authorization": f"Bearer {FIRECRAWL_KEY}",
                         "Content-Type": "application/json"},
                json={"url": full, "formats": ["rawHtml"], "onlyMainContent": False},
                timeout=60,
            )
            if r.status_code == 200:
                d = r.json()
                html = (d.get("data") or {}).get("rawHtml") or ""
                if html:
                    _fc_credits["pages"] += 1
                    return _Resp(200, html)
            # fall through to direct on non-200 or empty payload
        except Exception:
            pass
    return get(url, params=params)


def query(params):
    r = get(API, params=params); r.raise_for_status(); return r.json()


FIELDS = {
    "request[fields][ratings]": 1, "request[fields][active_installs]": 1,
    "request[fields][support_threads]": 1, "request[fields][support_threads_resolved]": 1,
    "request[fields][last_updated]": 1, "request[fields][added]": 1,
    "request[fields][tags]": 1, "request[fields][description]": 0,
    "request[fields][sections]": 0, "request[fields][icons]": 0,
    "request[fields][banners]": 0, "request[fields][screenshots]": 0,
}


def pull_popular():
    out = {}
    for page in range(1, POPULAR_PAGES + 1):
        d = query({"action": "query_plugins", "request[browse]": "popular",
                   "request[per_page]": 250, "request[page]": page, **FIELDS})
        for p in d.get("plugins", []):
            out[p["slug"]] = (p, "popular")
        print(f"popular p{page}: {len(d.get('plugins', []))}", flush=True)
        time.sleep(SLEEP)
    return out


def pull_tag(tag):
    out = {}
    for page in range(1, TAG_PAGES.get(tag, 1) + 1):
        d = query({"action": "query_plugins", "request[tag]": tag,
                   "request[per_page]": 250, "request[page]": page, **FIELDS})
        plugins = d.get("plugins", [])
        for p in plugins:
            out[p["slug"]] = (p, tag)
        print(f"tag {tag} p{page}: {len(plugins)}", flush=True)
        time.sleep(SLEEP)
        if not plugins:
            break
    return out


def author_slug(p):
    m = re.search(r"profiles\.wordpress\.org/([^/\"]+)", p.get("author_profile", "") or p.get("author", ""))
    return m.group(1).lower() if m else ""


def metrics(p, source):
    rt = {int(k): v for k, v in (p.get("ratings") or {}).items()}
    n = sum(rt.values())
    th = p.get("support_threads") or 0
    rs = p.get("support_threads_resolved") or 0
    lu = (p.get("last_updated") or "")[:10]
    try:
        stale = (datetime.now(timezone.utc) - datetime.strptime(lu, "%Y-%m-%d").replace(tzinfo=timezone.utc)).days
    except ValueError:
        stale = None
    a = author_slug(p)
    return {
        "slug": p["slug"],
        "name": re.sub(r"&#8211;|&amp;", "-", p.get("name", ""))[:60],
        "source": source,
        "author": a,
        "ownership": "captive/rollup" if a in CAPTIVE_AUTHORS else "indie?",
        "installs": p.get("active_installs") or 0,
        "avg": round((p.get("rating") or 0) / 20, 2),
        "reviews": n,
        "one_star_pct": round(100 * rt.get(1, 0) / n, 1) if n else None,
        "low_pct": round(100 * (rt.get(1, 0) + rt.get(2, 0)) / n, 1) if n else None,
        "five_star_pct": round(100 * rt.get(5, 0) / n, 1) if n else None,
        "threads_2mo": th,
        "unresolved_pct": round(100 * (th - rs) / th, 1) if th else None,
        "last_updated": lu,
        "days_stale": stale,
        "added": (p.get("added") or "")[:10],
        "tags": ",".join((p.get("tags") or {}).keys()) if isinstance(p.get("tags"), dict) else "",
    }


REL = re.compile(r"(\d+)\s+(year|month|week|day|hour|minute)s?")


def rel_to_days(text):
    days = 0.0
    for num, unit in REL.findall(text):
        num = int(num)
        days += num * {"year": 365, "month": 30, "week": 7, "day": 1, "hour": 1 / 24, "minute": 1 / 1440}[unit]
    return days


def one_star_reviews(slug, max_pages=REVIEW_PAGES):
    ages = []
    for page in range(1, max_pages + 1):
        url = f"https://wordpress.org/support/plugin/{slug}/reviews/"
        r = fetch_html(url, params={"filter": 1, "page": page})
        if r.status_code != 200:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.select("li.bbp-topic-freshness")
        if not items:
            break
        for it in items:
            t = it.get_text(" ", strip=True)
            if "ago" in t:
                ages.append(rel_to_days(t))
        time.sleep(SLEEP)
    return ages


def advanced_page(slug):
    r = fetch_html(f"https://wordpress.org/plugins/{slug}/advanced/")
    out = {"dl_yesterday": None, "dl_7d": None, "dl_all": None}
    if r.status_code != 200:
        return out
    soup = BeautifulSoup(r.text, "html.parser")
    txt = soup.get_text(" ", strip=True)
    for key, label in (("dl_yesterday", "Yesterday"), ("dl_7d", "Last 7 Days"), ("dl_all", "All Time")):
        m = re.search(label + r"\s+([\d,]+)", txt)
        if m:
            out[key] = int(m.group(1).replace(",", ""))
    return out


def version_split(slug):
    try:
        r = get("https://api.wordpress.org/stats/plugin/1.0/", params={"slug": slug})
        if r.status_code == 200:
            d = r.json()
            top = sorted(d.items(), key=lambda kv: -float(kv[1]))[:4]
            return "; ".join(f"{k}:{float(v):.1f}%" for k, v in top)
    except Exception:
        pass
    return ""


def pre_score(r):
    """Cheap ranking proxy (no scraping) to choose which candidates to enrich."""
    est_one_star = (r["reviews"] or 0) * ((r["one_star_pct"] or 0) / 100)
    weight = 1.0 if r["ownership"] == "indie?" else 0.3
    return est_one_star * math.sqrt((r["installs"] or 0) / 100_000) * weight


ENRICH_KEYS = ["one_star_scraped", "one_star_last_12mo", "one_star_12_24mo",
               "one_star_recent_share_pct", "dl_yesterday", "dl_7d", "dl_all",
               "version_split", "polarized", "pre_score", "enriched", "seam_score"]


def main():
    pulled = pull_popular()
    for tag in TAGS:
        for slug, v in pull_tag(tag).items():
            pulled.setdefault(slug, v)

    rows = [metrics(p, src) for p, src in pulled.values()]
    rows.sort(key=lambda r: -r["installs"])
    with open(f"{OUT}/all_plugins.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} unique plugins pulled", flush=True)

    cands = [r for r in rows if r["installs"] >= MIN_INSTALLS and r["reviews"] >= MIN_REVIEWS
             and (r["avg"] < MAX_AVG or (r["one_star_pct"] or 0) >= MIN_ONE_STAR_PCT)]
    for r in cands:
        r["pre_score"] = round(pre_score(r), 1)
        for k in ENRICH_KEYS:
            r.setdefault(k, None)
        r["enriched"] = False
    cands.sort(key=lambda r: -r["pre_score"])
    to_enrich = cands[:ENRICH_TOP]
    print(f"{len(cands)} candidates; enriching top {len(to_enrich)} (bounded)...", flush=True)

    for i, r in enumerate(to_enrich, 1):
        ages = one_star_reviews(r["slug"])
        r["one_star_scraped"] = len(ages)
        r["one_star_last_12mo"] = sum(1 for a in ages if a <= 365)
        r["one_star_12_24mo"] = sum(1 for a in ages if 365 < a <= 730)
        r["one_star_recent_share_pct"] = round(100 * r["one_star_last_12mo"] / len(ages), 1) if ages else None
        r.update(advanced_page(r["slug"]))
        r["version_split"] = version_split(r["slug"])
        r["polarized"] = bool(r["five_star_pct"] and r["one_star_pct"]
                              and r["five_star_pct"] > 35 and r["one_star_pct"] > 25)
        r["enriched"] = True
        recent = r["one_star_last_12mo"] or 0
        r["seam_score"] = round(recent * (r["installs"] / 100_000) ** 0.5
                                * (0.3 if r["ownership"] != "indie?" else 1.0), 1)
        print(f"  [{i}/{len(to_enrich)}] {r['slug']}: 1* last12mo={r['one_star_last_12mo']} / {len(ages)} score={r['seam_score']}", flush=True)
        time.sleep(SLEEP)

    cands.sort(key=lambda r: (-(r["seam_score"] or -1), -r["pre_score"]))
    fieldnames = list(rows[0].keys()) + ENRICH_KEYS
    with open(f"{OUT}/candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(cands)

    enriched = [r for r in cands if r["enriched"]]
    with open(f"{OUT}/summary.md", "w") as f:
        f.write(f"# WordPress plugin candidate summary\n\n")
        f.write(f"- Plugins pulled: {len(rows)}\n- Candidates: {len(cands)}\n")
        f.write(f"- Enriched (top by pre-score): {len(enriched)}\n\n")
        f.write("| score | slug | owner | installs | avg | 1*% | 1* last 12mo | scraped | unresolved% | versions | polarized |\n")
        f.write("|--|--|--|--|--|--|--|--|--|--|--|\n")
        for r in enriched:
            f.write(f"| {r['seam_score']} | {r['slug']} | {r['ownership']} | {r['installs']:,} | {r['avg']} | "
                    f"{r['one_star_pct']} | {r['one_star_last_12mo']} | {r['one_star_scraped']} | "
                    f"{r['unresolved_pct']} | {r['version_split']} | {r['polarized']} |\n")
    if FIRECRAWL_KEY:
        print(f"Firecrawl: {_fc_credits['pages']} pages fetched (~{_fc_credits['pages']} credits)", flush=True)
    print(f"\nDone. See {OUT}/summary.md", flush=True)


if __name__ == "__main__":
    main()
