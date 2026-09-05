# Fact-check of every claim in the offer, landing page, and research docs (2026-09-05)

Three verification passes against primary sources, plus an anti-slop audit
(github.com/miqdadbadjuber/anti-slop, AFTER mode) of the landing page and report.
Verdicts: HOLDS, PARTIAL, FAILS. Corrections have been applied to the docs and the page.

## Mindbody

| Claim | Verdict | What the source says | Action |
|---|---|---|---|
| Capterra 4.0 / 2,994 reviews; value 3.6, ease 3.9, support 3.8 | HOLDS | Capterra product page, checked Sep 2026 | Keep, date-stamp |
| Lowest value-for-money in its Capterra categories | HOLDS | 3.6 vs 3.7 next lowest (Virtuagym) in coaching, yoga, personal-trainer | Keep, phrase as "lowest of any product listed" |
| 388 one- and two-star reviews | PARTIAL | Now 249 + 142 = 391, 13% | Use "13%" |
| $499 export fee to leave | PARTIAL | ToS 8.3 and Mindbody's blog: standard CSV exports free; large or encrypted card-data export is a paid service, fee not published; third parties report ~$499-500 | "About $500, reported, for the stored-card export; client exports free" |
| Annual auto-renew, no early exit, $1,200/yr | PARTIAL | Mindbody blog: 12/24/36-month terms, auto-renew, 30 days' notice, no mid-term cancellation; Starter listed at $79/mo; $1,200 is one reviewer's contract | Use "12 to 36 months"; drop $1,200 as a general figure |
| Long-time customers pay 50%+ more | FAILS | Escalation widely reported (BBB Jan 2026, Capterra Aug 2025 "over $1,000 CAD/month") but no source states 50% | Removed from page |
| Mindbody Payments powered by Stripe | HOLDS | mindbodyonline.com/business/payments | Keep |
| ~3.5% processing, debit same as credit | PARTIAL | Mindbody publishes no rate; third parties report 2.99% + 30c in person, 3.60% + 30c online; 3.5% + 20% applies to marketplace sales | Page now says rate is unpublished and computed from invoices |
| 60,000+ businesses | FAILS | Mindbody's site: "more than 40,000" | Corrected to 40,000+ |
| Cancellation problems, 2025-26 | HOLDS | BBB Dec 2025, Feb 2026, Mar 2026, Aug 2026; Trustpilot Jul 2026 | Keep, summarised without quotation marks |

## Alternatives and card migration

| Claim | Verdict | What the source says | Action |
|---|---|---|---|
| Vagaro 4.7 / 3,651; from $23.99 | PARTIAL | Rating holds; $23.99 is a 6-month promo, standard $30 | Page says $30 |
| Vagaro free Mindbody import includes cards | FAILS | Vagaro help article lists cards on file and notes as "Contact Mindbody" | Page says none of the three moves cards |
| WellnessLiving 4.4 / 613; $69-349; free migration + onboarding specialist | HOLDS | Vendor pricing page and help centre | Keep |
| WellnessLiving migrates stored cards | FAILS | Help article: payment methods "cannot be imported"; sales history and auto-assigned memberships also not migrated | Page and FAQ corrected |
| WellnessLiving contract | PARTIAL | Vendor does not publish term; customers report 12-month auto-renew | Page says "annual term reported" |
| TeamUp 4.8 / 346; $119; month to month; Mindbody import | HOLDS | Vendor pages; card import only from a Stripe/GoCardless account the studio owns | Keep |
| Momence: Clubessential Jan 2025, Xplor Mar 2026 | HOLDS | Xplor press release; PR Newswire Mar 30, 2026 | Keep |
| Momence hidden surcharges since acquisition | PARTIAL | Pricing is published (Basic $0, Pro $60, Custom $199, 3.9% + 30c); billing complaints on Capterra Apr 2025, Sep 2025, Apr 2026; no causal link to the acquisition | Dropped "hidden" and "since acquisition" |
| Glofox acquired by ABC 2023 | PARTIAL | Completed Aug 25, 2022; vendor publishes only "from $99/mo" | Corrected |
| Walla floor $320, Pro $599 | HOLDS | hellowalla.com/pricing | Keep |
| Arketa migration complaints | HOLDS | Capterra Jun 2025 to Aug 2026: "data migration was a complete disaster", billing data withheld on exit; plus two r/pilates threads from our Arctic Shift pull | Keep, attributed |
| Stripe moves stored cards processor to processor | PARTIAL | Stripe PAN import/export exist, PCI Level 1 destination, 2-6 weeks; only the Stripe account owner can request it, and for Mindbody Payments that owner is Mindbody | Offer now has three cases: Mindbody paid export, TeamUp own-Stripe import, or re-card campaign |
| Vendors pay consultant referrals | PARTIAL | WellnessLiving: $1,500 cash per referred studio (partner program); Vagaro: $50 credit to existing customers; TeamUp: gift card to customers | Disclosure line added to page and report |

## Legal, consumer, model

| Claim | Verdict | What the source says | Action |
|---|---|---|---|
| NYC click-to-cancel effective Oct 1, 2026 | HOLDS, but consumer-only | Adopted Jul 10, 2026; sits under NYC consumer protection law (personal, household, family purposes); a studio's Mindbody plan is a business purchase | Do not cite it for B2B; removed from offer positioning |
| FTC click-to-cancel vacated 2025 | HOLDS | 8th Circuit, Jul 2025; FTC restarted rulemaking Mar 2026 | Note only |
| CR: junk fees $3,200 per family | PARTIAL | CR petition to FTC, Jan 2023, citing 2019 CR research; no methodology page | Cite as a CR estimate; not used on the studio page |
| CR mental-health app privacy study, Mar 2021 | HOLDS | Digital Lab Dec 2020, article Mar 2021 | N/A, lead dropped |
| Corey Ganim $999 assessment, 50-60% upsell, $8K MRR in 10 days | HOLDS | Startup Ideas Podcast, Jul 13, 2026; "50%"; all self-reported | Use "roughly half" |
| Cal AI $40M, 250 influencers, MyFitnessPal | HOLDS | TechCrunch Mar 2026: >$30M annual revenue; Inc: $40M; closed Dec 2025 | Cite TechCrunch's $30M |
| 3.5% of apps reach $10K MRR; median 18-36 months | FAILS | RevenueCat 2026: 4.6% reach $10K monthly revenue within 2 years; median 109 days for those that do; iOS + Android | Corrected in leads doc |
| FamilyControls approvals 10-14+ days, spring 2026 | HOLDS | Apple forum threads Mar-Apr 2026; DTS: "working to clear the backlog" | Keep |
| Opal $17.1M ARR, 1M DAU, two-thirds students | PARTIAL | Founder (Apr 2026): 1M DAU, $10M ARR, two-thirds students; $17M is a Latka estimate | Corrected |
| Arctic Shift needs one subreddit per text search | HOLDS | API README | Keep |

## Anti-slop audit, applied

Hard-gate items fixed: every number on the page now has a source line and a check date;
paraphrased Reddit quotes replaced with an attributed summary list; the stored-card promise
rewritten as three cases; vendor prices dated; "Medium" risk colour changed to #75681a
(5.6:1); Calendly placeholder flagged visibly. Purpose-gate and quality items fixed:
checkmarks replaced with serif numerals; three-card pricing replaced with a ruled list with
no highlighted tier; a written reason for uppercase eyebrows added to the stylesheet;
focus-visible outline added; refund wording unified between hero and guarantee; the run of
"X, not Y" sentences reduced; report template placeholders no longer pre-fill vendor names
with allegations.
