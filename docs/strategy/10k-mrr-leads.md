# Three Leads to $10K MRR, sourced from Capterra, G2, Consumer Reports, Google Trends, and Reddit (Arctic Shift)

Researched 2026-09-05. Primary sources: Capterra and G2 (review volume, sub-scores, 1-star
themes), Consumer Reports (privacy, junk fees, cancellation), Google Trends (US, 12-month and
5-year, pulled directly with `trends_probe.py`). Reddit was pulled directly through the
Arctic Shift API with `reddit_arctic_shift.py` (posts and comments since 2025-09-01).

**Scope rule from the founder: no mental-health or medical positioning.** ADHD, therapy,
sleep, and anything that implies treatment are out on compliance grounds. All three leads
below are business tooling or productivity.

## What the sources say, in one table

| Signal | Source | Number |
|---|---|---|
| Mindbody overall / value-for-money | Capterra, 2,994 reviews | 4.0 / 3.6 (lowest in category) |
| Mindbody 1-2 star reviews | Capterra | 388 (13%) |
| Mindbody cost complaints | G2 | "most frequently cited concern"; prohibitive for small studios |
| Reddit posts mentioning Mindbody, studio-owner subs, 12 mo | Arctic Shift | 157 posts, 506 comments |
| ...of which about switching / leaving | Arctic Shift | 54 posts, 204 comments |
| ...of which about price / fees | Arctic Shift | 88 posts, 231 comments |
| "mindbody alternatives" search | Google Trends, rising 12-mo | +250% |
| "mindbody pricing" search | Google Trends, rising 12-mo | +190% |
| "screen time app" search, 5-year | Google Trends | 8.9 to 37.0 (4x), still rising |
| Reddit posts mentioning a screen-time app, 12 mo | Arctic Shift | 739 |
| ...bypass complaints / asking for a recommendation | Arctic Shift | 95 / 196 |
| Reddit "lock in" / study-together posts, 12 mo | Arctic Shift | 979 |
| ...asking for a partner or group | Arctic Shift | 341 |
| "locked in" search, 12-mo | Google Trends | 56 to 70 |
| "study app" search, 12-mo | Google Trends | 13 to 20 |
| Junk fees per family per year | Consumer Reports | $3,200 |
| "brain rot", "sleepmaxxing", "ugc ads" search | Google Trends | dead or ~0 |

Trends terms in one group are relative to each other. Anchored in one group:
`adhd` 79, `screen time app` 4.9, `mindbody` 2.0, `app blocker` 0.9, `personal trainer app` 0.
Arctic Shift's most recent 4-6 weeks are under-counted (ingestion lag), so treat the last
two months of every "by month" series as a floor.

---

## Lead 1 (cash in weeks): "Mindbody Exit" done-for-you migration service for one-location studios

**Capterra / G2.** Mindbody is the lowest-rated high-volume product in every Capterra
category it appears in: 4.0 overall, 3.6 value, 3.8 support, 3.9 ease of use, 2,994 reviews,
388 of them 1 or 2 stars. Reviewers are owner-operators of single-location Pilates, barre,
yoga, dance, and martial-arts studios. Their words: "$1,200, 1-year contract, their sales
people appear to be on commission"; bills from ~$99/mo to $699+/mo per location before
add-ons; "over $1,000 CAD/month, almost impossible to get real help"; a paid export for stored card data, reported at about $500, that appears only when you leave (standard client exports are free). G2: cost is the most-cited complaint, with hidden
fees and high processing rates.

**Google Trends.** "mindbody" brand search is flat for 5 years at a large volume, and every
rising query is exit intent: "mindbody alternatives" +250%, "mindbody pricing" +190%,
"mindbody reviews" +100%, competitor "vagaro" +300%. Mindbody's site says more than 40,000 businesses.

**Reddit, pulled directly.** 157 posts and 506 comments in r/mindbody, r/gymowner,
r/FitnessStudioOwner, r/pilates, r/YogaTeachers, r/personaltraining since Sept 2025.
Comment volume is rising: 19 in Sept 2025, 90 in May 2026, 65 in July 2026. Top threads:
"Anyone switched away from Mindbody?", "Thinking of moving off of Mindbody", "Has anyone
migrated away from Mindbody?", "Mindbody outage got me thinking about switching",
"Arketa - What a nightmare", "Arketa Migration - Not what they promised". Quotes:
- "Mindbody makes it almost impossible to leave. Everything is tied to..."
- "I plan to cancel their recurring memberships after we migrate, but I'm worried about the stored card data. Do I need to manually clear every client's payment info?"
- "Was on the phone with a specialist who scheduled the downgrade and a cancellation... they have no record of a downgrade"
- "Last week's outage was like 6+ hours... already annoyed with the price and support, this pushed me over"
- "We're paying over 50% more right now than when we first started, with less service"
- "When a client cancels a class from a package that was transferred over, the credit does not get returned" (a botched migration)
Also notable: vendors (Vibefam, Time2book, YogaCRM) are already astroturfing these
threads, which tells you they will pay for referred studios.

**The offer.**
| Package | Price | What you do |
|---|---|---|
| Exit Audit | $199 | Read the contract, find the renewal window, compute true cost, pick the target platform |
| Full Migration | $1,200 to $1,800 | Export members/packages/memberships, rebuild schedule and pricing, coordinate the Stripe token transfer (Mindbody Payments runs on Stripe), run a parallel week, cut over, verify package credits |
| Retainer | $149/mo | Reporting dashboard (the #1 non-price complaint) and a monthly check-in |

Six migrations a month is $7K to $10K before retainers and vendor referrals.

**Where the first 10 customers are.** The 54 Reddit posts above are people asking to be
sold this, by name. Capterra 1-star reviewers are public with role and business type.
$5/day Google Ads on "mindbody alternatives" and "mindbody pricing". Ask Vagaro,
WellnessLiving, Vibefam, Momence, and Glofox for a referral deal on day one; none publish
one, all have sales teams hunting these studios.

**30-day plan.** Days 1-5: one free migration for a local studio to build the checklist and
testimonial. Days 6-15: 20 touches a day, sell the $199 audit. Days 16-30: convert audits
to migrations. No paid audit by day 14 means the price or channel is wrong.

**Risks.** Live billing is involved; always run the parallel week and verify package
credits (the exact failure in the Arketa threads). Signed scope per studio.

---

## Lead 2 (ship now, no Apple gate): "Lock In" social study sessions for students

**Google Trends.** "locked in" rose from 56 to 70 over 12 months and "study app" from 13
to 20. Both are Gen Z framing for the same thing: sitting down and doing the work.

**Reddit, pulled directly.** 979 posts in r/GetStudying, r/studytips, r/getdisciplined,
r/college, r/productivity since Sept 2025 about locking in or studying together, 341 of
them explicitly asking for a partner or group and 263 naming the phone as the problem.
This is the highest-engagement corpus in the whole research: "Lock in with me!" (+1,583),
"that's it. i'm locking in" (+1,558), "Saw this, and it has become my motivation to lock
in for this semester" (+4,761), "Day 2 - trying to study 10 hours a day" (+1,284, 151
comments). Quotes:
- "Create some discord group / community for people DEDICATED TO STUDY, where we can do something together" (+784)
- "I'll hop on studystream or similar platforms where people are just studying together. It's weirdly motivating" (+225)
- "Willing to pay if it's not too expensive and actually works (and preferably has a student discount!). $30 a month with no student discount is a joke" (+453)
- "I'm wondering if I'd be better off studying [alone]... it's easier to complain together than to sit quietly and grind" (+185)
Volume peaks Sept to Nov (semester start). It is September now.

**Capterra / G2.** No coverage; this is a consumer category. The nearest B2C comps are
StudyStream (free, web), Focusmate ($6.99, 1:1 video, must book), Flow Club ($19-25), and
Forest. None of them is a phone-native "lock in with your friends" room with a streak.

**Consumer Reports.** Use the junk-fee and cancellation findings as the pricing promise:
monthly-first, visible cancel, real student price.

**The wedge.** Start or join a "lock in" room with friends or strangers, timer plus
camera-optional presence, phone-down detection using the app's own foreground state (no
FamilyControls needed), streaks and a shareable "hours locked in this week" card for
TikTok. Free for 2 sessions a day, $3.99/mo or $24.99/yr with a verified student price.
$10K MRR is roughly 2,500 monthly subscribers, which is one viral semester-start post.

**30-day plan.** Week 1: SwiftUI room + timer + presence, RevenueCat paywall. Week 2:
TestFlight to 30 people recruited from the 341 "looking for a partner" posters. Week 3:
App Store submit, daily "lock in with me" TikToks using the app's own share card.
Week 4: campus Discord seeding, student-discount landing page.

**Risks.** Seasonal; expect a dip in December and May. Group video costs money; keep
presence lightweight (audio-off, low-fps) and cap free rooms.

---

## Lead 3 (biggest structural demand, gated by Apple): screen-time blocker that survives bypass and is honest about billing

**Google Trends.** "screen time app" is up 4x over 5 years and still rising over the last
12 months while every meme term ("brain rot" 93 to 18, "dopamine detox") collapsed. Rising
queries are intent: "how to reduce screen time" +170%, "app to reduce screen time" +90%,
"best screen time app" +70%.

**Reddit, pulled directly.** 739 posts mentioning a screen-time app (Opal, ScreenZen,
one sec, Brick, Forest, etc.) in r/digitalminimalism, r/iphone, r/nosurf, r/studytips,
r/getdisciplined since Sept 2025: 196 asking for a recommendation, 108 about price or
cancellation, 95 about bypassing, 170 mentioning a friend or accountability. Quotes:
- "0/10. Disgustingly easy to bypass. Almost impressed with my ability to hit '15 more minutes' twelve times in a row" (+887, the top-rated review post in r/digitalminimalism)
- "The problem was simple: I could always bypass the restrictions myself" (+248)
- "I got Opal about a month ago; however, I've realized I could've saved $100 by just having a friend lock my Screen Time limits with a passcode of their own" (+50)
That last one is the product spec.

**App Store 1-star reviews.** Two piles: trivially bypassed (delete the app, change the
date, tap ignore) and trials that auto-convert to $99/yr with a hidden cancel. An
independent Oct 2025 to Apr 2026 test had teenagers bypassing 12 of 18 blockers within 48
hours.

**Consumer Reports.** Junk fees cost a family $3,200 a year; New York City's click-to-cancel
rule takes effect October 1, 2026. "Cancel in one tap, no annual auto-convert" is a
feature the category leader does not have.

**The wedge.** A friend holds the key. You set the block, your friend's passcode is the only
way to unlock early, and they get pinged when you try. $4.99/mo or $29.99/yr, monthly-first,
visible cancel. Opal proved the student freemium motion (two-thirds of its 1M daily users
are students) and the price gap between free ScreenZen and $19.99 Opal is open.

**Why it is third.** Blocking requires Apple's FamilyControls entitlement, and in 2026
indie developers report 10 to 14+ day waits with no reply, then a second wait for the
Shield extension. Submit the request on day 1. Lead 2 ships while you wait, and Lead 2's
users are Lead 3's first users.

---

## Dropped, with the source that killed it

- **ADHD / body-doubling / any mental-health app.** Founder decision: compliance. The demand
  is real (Trends, Reddit) but Consumer Reports' privacy findings and the health-data
  rules that follow make it the wrong first product.
- **AI UGC ad service.** "ugc ads" and "ai ugc" are ~1 on Trends. No inbound demand.
- **Sleep / sleepmaxxing.** Trends 0, hardware-led, and medical-adjacent.
- **"Brain rot" positioning.** 93 to 18 in 12 months.
- **Personal trainer or tutoring software.** Capterra shows the categories well served
  (PT Distinction 4.9, TrueCoach 4.8, Acuity 4.8); Trends for "personal trainer app" is 0.

## How to keep using these sources

- **Capterra.** Category pages expose value-for-money and ease-of-use sub-scores; a
  high-volume product with a value score 0.4 below its overall is a switching market.
  Review pages list reviewer role and business type: that is a lead list.
- **G2.** Blocks scrapers (403). Read search-indexed "what do you dislike" snippets and the
  compare pages; nearly everything scores above 4.0 so read dislikes, not scores.
- **Consumer Reports.** Thin on apps. Use it for trust and pricing claims, not demand.
- **Google Trends.** `trends_probe.py` (pytrends, `retries=0`). Anchor term in every group,
  5-year line for structure, 12-month line for momentum, rising queries for intent.
- **Reddit via Arctic Shift.** `reddit_arctic_shift.py`. Text search only works with a single
  `subreddit` per request; comments use `body=`, posts use `query=`. Recent weeks lag.
  Vendors astroturf the switching threads; discount comments that name a product in the
  first sentence.

## Sequence

| Week | Lead 1 (Mindbody Exit) | Lead 2 (Lock In app) | Lead 3 (blocker) |
|---|---|---|---|
| 1 | Free pilot migration, checklist, testimonial | Room + timer + paywall | Submit FamilyControls request |
| 2 | 20 touches/day, sell $199 audits | TestFlight with 30 Reddit recruits | Wait |
| 3 | First paid migrations | App Store submit, daily lock-in TikToks | Wait |
| 4 | 4-6 migrations booked ($5-9K) | Student-discount page, campus Discords | Build if approved |
