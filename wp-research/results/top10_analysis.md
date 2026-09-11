# Top 10 by seam_score, with complaint themes (snapshot 2026-09-11)

Numbers below come from the corrected v3 run (fixed pagination, verified
ownership, exact recent counts). Complaint themes are summarised from the titles
of 1-star reviews posted in the last 12 months (`one_star_titles.json`). Titles
are the reviewers' own words, not verified facts.

## Headline finding

With pagination fixed, **no independently-owned plugin in the candidate set has
more than 7 one-star reviews in the past 12 months.** The large recent-complaint
volumes reported by earlier runs were an artifact of re-reading page 1. The real
"seams" are almost all platform-owned: forced installs, consent violations,
bait-and-switch pricing, and breaking updates by SiteGround, Automattic, Elementor,
Meta and Stripe. The scoring formula also over-weights installs: Contact Form 7
ranks 7th on a single complaint because of its 10M installs.

## Top 10

| # | plugin | owner | installs | 1★ last 12mo | what the 1-star reviews complain about |
|--|--|--|--|--|--|
| 1 | sg-ai-studio | SiteGround (captive) | 1.0M | 81 | Force-installed on customer sites without consent (one reviewer: 200+ client sites); new ToS said to grant SiteGround co-ownership of AI-generated content; repeatedly called malware. |
| 2 | jetpack | Automattic (captive) | 3.0M | 15 | Version 16.1 and Jetpack Boost breaking or crashing sites; Elementor editing errors; stats moved behind a paywall; forms update described as a mess. |
| 3 | google-listings-and-ads | Automattic/Woo (captive) | 800k | 23 | Cannot connect or finish setup; Merchant Center sync fails or loops; field mismatches breaking admin; an update crashing many sites. |
| 4 | image-optimization | Elementor (captive) | 1.0M | 19 | Bait-and-switch pricing: every generated image size counts against the 200 free quota; non-refundable subscriptions; slow, poor quality, breaks the post editor. |
| 5 | woocommerce-paypal-payments | WooCommerce (captive) | 800k | 14 | Update 4.0.3 broke sites; plugin auto-enabled eight unwanted payment methods without consent; general instability for a payment-critical plugin. |
| 6 | pojo-accessibility (Ally) | Elementor (captive) | 500k | 15 | "It was good, now Elementor bought it": stealth Elementor menus and upsells, new paywall, GDPR non-compliance, welcome-page lockups. Classic acquired-and-ruined pattern (56 more 1★ in the prior year). |
| 7 | contact-form-7 | Rock Lobster (indie, verified) | 10M | 1 | One report of a quiz field loading intermittently. No seam; the rank is pure install weight. |
| 8 | facebook-for-woocommerce | Meta (captive) | 400k | 14 | Site crashes and fatal errors; setup fails silently when third-party cookies are blocked; Conversions API not working. |
| 9 | woocommerce-gateway-stripe | Stripe/Woo (captive) | 700k | 10 | Updates silently turn features on ("defaced our checkout without consent"); regressions every release; support unhelpful. |
| 10 | gutenberg | WordPress core (captive) | 400k | 13 | Generic dislike of the block editor UX. Not actionable. |

## Highest-scoring independent plugins

| plugin | owner (verified) | installs | 1★ last 12mo | complaint themes |
|--|--|--|--|--|
| woocommerce-multilingual | OnTheGoSystems / WPML | 100k | 7 | Slow ("to slow down your site install WPML"), buggy, expensive, auto-renewed without warning; support acknowledged as responsive. |
| jeg-elementor-kit | Jegtheme | 300k | 3 | Update process called intentionally deceptive and convoluted; nagging behaviour. |
| google-captcha | BestWebSoft | 100k | 3 | "Doesn't work" (three near-identical titles). |
| nitropack | NitroPack | 90k | 2 | Billing dispute, "useless". |
| email-address-encoder | Till Krüss / CacheWerk | 100k | 1 | Unanswered support. |

## Other observations

- **backwpup** (WP Media, captive): 151 one-star reviews 12-24 months ago, down to 6
  in the last year. That was the v5 rewrite backlash after the WP Media acquisition;
  the seam has largely closed.
- **wp-user-avatar** (ProfilePress, reclassified captive): 1 recent complaint, but the
  titles across its history ("Sneaky bait & switch", "Plugin hijacked", "Not the same
  plugin") document the acquisition-and-rebrand pattern the `polarized` flag targets.
- The `polarized` flag fires mostly on acquired or platform plugins
  (pojo-accessibility, image-optimization, wp-user-avatar, woocommerce-paypal-payments).
- Ownership corrections moved 23 candidate rows from `indie?` to captive/rollup; see
  `ownership_verification.md`. The earlier "indie" top list (backwpup, pagelayer,
  contact-form-7-honeypot, wp-user-avatar) was entirely rollup-owned.

## Scoring caveat

`seam_score = recent_1★ × sqrt(installs / 100k) × (0.3 if captive else 1.0)`. The
install term dominates at the top (Contact Form 7, 10M installs, 1 complaint, rank 7).
For an indie-focused shortlist, rank on `one_star_last_12mo` with a minimum floor
(e.g. ≥5) rather than on `seam_score`.
