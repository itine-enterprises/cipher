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

## Parser fixes (v2)

- **Download history** now comes from the stats API
  (`stats/plugin/1.0/downloads.php?slug=…&historical_summary=1`). The plugin
  "Advanced" page renders those numbers client-side; its HTML only carries the
  localisation labels, which is why the original text scrape returned nothing.
- **Review pagination stops early.** Reviews list newest-first, so the scraper
  now stops once a whole page is older than 24 months instead of at a fixed page
  count. `one_star_last_12mo` and `one_star_12_24mo` are therefore exact counts,
  not capped at the page limit (v1 saturated at 360 for busy plugins). The
  40-page limit is only a safety cap.
- **Review titles are captured** for the recent (≤12mo) 1-star reviews:
  `one_star_titles_recent` in `candidates.csv` (first 40) and the full lists in
  `results/one_star_titles.json`.

## Ownership verification

`ownership` is a lookup against the `CAPTIVE_AUTHORS` set. In v2 every `indie?`
candidate with ≥30k installs was checked against its live profile page; 21 authors
(23 plugin rows) turned out to be platforms, big-tech, hosting-owned, or serial
acquirers and were moved to captive/rollup. Several v1 entries also used the
wrong slug (`wpmedia` vs the real `wp_media`, `brevo` vs `neeraj_slit`,
`microsoft` vs `bingwebmastertools`, `wpdeveloper` vs `wpdevteam`). The full
per-author evidence is in `results/ownership_verification.md`. Remaining
`indie?` rows below 30k installs are unverified.

## Caveats (read before using the numbers)

- **`ownership` is still a heuristic below 30k installs.** Above that threshold
  it has been verified (see above); below it, "indie?" means "not on the list."
- **`seam_score` is a ranking heuristic**, not a measurement: recent 1-star volume x
  sqrt(installs/100k), down-weighted 0.3x for captive/rollup owners.
- **Only the top `ENRICH_TOP` candidates by pre-score are enriched.** Candidates
  outside that set have blank enrichment columns and no `seam_score`.
- **1-star review titles** are the reviewer's own words and are provided as data
  for theme analysis, not as verified facts about the plugin.

Snapshot date: 2026-09-11.
