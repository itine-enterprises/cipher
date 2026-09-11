# WordPress plugin research dataset

Aggregates **public** WordPress.org data into a market-research dataset: for each
plugin, the install count, star breakdown, support-thread resolution rate, staleness,
and (for the top candidates) the recency of 1-star reviews and active-version split.

Data sources (all public, no auth):
- Plugin info API: `https://api.wordpress.org/plugins/info/1.2/`
- Version stats API: `https://api.wordpress.org/stats/plugin/1.0/`
- Public review pages: `https://wordpress.org/support/plugin/<slug>/reviews/?filter=1`

## Reproduce

```
pip install requests beautifulsoup4
python wp_seam_miner.py      # writes to ./results/
```

The script is polite to wordpress.org: 1.5s between requests, backoff on HTTP 429,
and the review scrape is bounded (see below).

### Optional: route HTML scrapes through Firecrawl

The two HTML scrapes (review pages, advanced page) can go through
[Firecrawl](https://firecrawl.dev) instead of direct requests, so a larger run
doesn't get this IP throttled. Set the API key and run:

```
export FIRECRAWL_API_KEY=fc-...      # free tier is ~500-1000 credits/month
python wp_seam_miner.py
```

When the key is unset it falls back to direct requests (what the committed
dataset used). Notes:
- The JSON APIs (plugin info, version stats) never use Firecrawl — they are
  plain APIs, not scraping, so routing them would waste credits.
- Firecrawl bills ~1 credit per page. The default bounded run is
  `ENRICH_TOP * REVIEW_PAGES` = 30 * 12 = ~360 page fetches, which fits the free
  tier. Enriching all 119 candidates at 12 pages (~1400 fetches) does not.
- For these specific pages Firecrawl does not improve data quality — the pages
  are plain server-rendered HTML that direct requests already parse. Its only
  benefit here is proxying to avoid rate limits at larger scale.
- `FIRECRAWL_API_URL` overrides the endpoint (default `.../v2/scrape`).

## Outputs (`results/`)

- `all_plugins.csv` — every plugin pulled (3299), with metrics.
- `candidates.csv` — the 119 plugins passing the candidate filter; the top 30 by
  pre-score also carry the scraped enrichment columns.
- `summary.md` — the 30 enriched candidates, ranked by `seam_score`.

## Method

1. **Pull** — popular pages 1-2 plus 12 category tags (250/page), deduped by slug.
2. **Candidate filter** — installs >= 10k, reviews >= 20, and (avg < 4.0 or 1-star% >= 15).
3. **Enrichment (bounded)** — only the top `ENRICH_TOP=30` candidates by a cheap
   pre-score are scraped, at `REVIEW_PAGES=12` pages each (~360 recent reviews max),
   to keep the total request volume modest. The original single-session script
   enriched all candidates at 40 pages; that is thousands of requests and is not
   run here.

## Caveats (read before using the numbers)

- **Review recency is capped.** For high-traffic plugins the 12-page cap only
  reaches the most recent ~360 1-star reviews, so `one_star_last_12mo` and
  `one_star_recent_share_pct` saturate (e.g. 360/360) and understate total lifetime
  1-star volume. Treat them as a "recent complaint intensity" signal, not a full count.
- **`dl_yesterday` / `dl_7d` / `dl_all` are blank.** The plugin "Advanced" page no
  longer exposes these as scrapable text; the selector returns nothing. Version
  split (from the stats API) is populated instead.
- **`ownership` is heuristic.** "captive/rollup" vs "indie?" is a lookup against a
  hand-maintained author list (`CAPTIVE_AUTHORS`); "indie?" means "not on the list,"
  not verified independent ownership.
- **`seam_score` is a ranking heuristic**, not a measurement: recent 1-star volume x
  sqrt(installs/100k), down-weighted 0.3x for captive/rollup owners.

Snapshot date: 2026-09-11.
