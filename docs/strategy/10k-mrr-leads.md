# Three Leads to $10K MRR (researched 2026-09-05)

Goal: fastest path to cash, then compound toward $10K MRR. Constraint: money is tight, so the
first lead has to pay bills in weeks, not months.

## The uncomfortable number first

Consumer iOS apps are slow money. Across tracked indie iOS apps, the median timeline is:

| Milestone | Typical timeline |
|---|---|
| First $100/mo | 1-3 months |
| $1K MRR | 4-12 months |
| $10K MRR | 18-36 months |

Only ~3.5% of apps ever reach $10K MRR. Services reach first revenue in 2-4 weeks.
So the plan is: **run a service for cash now, and use the same distribution machine to
launch an app.** The thread post you shared is right about distribution being the moat.
It is wrong about the timeline for a solo founder with bills due.

Ranking below is by *speed to first dollar*, not by ceiling.

---

## Lead 1 (do this first): AI UGC ad factory as a productized service

**What it is.** A "done-for-you" subscription that ships app founders and Shopify stores
20-40 short-form ad variants per month, made with AI actors and AI scripts, in TikTok /
Reels / Shorts formats. This is literally the "AI Influencer Factory" from the post,
sold to other people before you use it on your own app.

**Why it's the fastest money.**
- Human UGC creators charge $150-$212 per video on average, and $500-$1,200 for pros.
- AI UGC tools (Arcads, MakeUGC) produce a video for roughly $10-$25 of tool cost.
  Arcads Starter is $77-$110/mo; MakeUGC is $49-$119/mo.
- The buyer already knows they need volume. Cal AI (bootstrapped, $40M ARR, sold to
  MyFitnessPal) got there by paying 250 creators monthly retainers for organic-looking
  posts. Every indie founder has read that story and cannot afford it.
- Services get paid up front. No App Review, no trial-to-paid funnel, no RevenueCat wait.

**Offer and pricing (starting point).**
| Tier | Price | Deliverable |
|---|---|---|
| Starter | $497/mo | 20 ad variants, 5 hooks x 4 angles, captions, 2 revisions |
| Growth | $997/mo | 40 variants + weekly hook testing report + 1 AI persona built for the brand |
| Sprint | $297 one-off | 10 variants in 72 hours (foot-in-the-door offer) |

Ten Starter customers is $5K MRR. Tool cost per customer is under $150.
Twenty Starter/Growth customers gets past $10K MRR, and the work is mostly a pipeline.

**Where the first 10 customers are.**
- r/iOSProgramming, r/SaaS, r/shopify, r/Entrepreneur: search "UGC", "TikTok ads",
  "creative fatigue", "need more ad creatives". Reply with a free 3-variant sample.
- X "build in public" accounts posting MRR screenshots between $1K and $20K MRR. They
  have budget and no creative team. DM 20 per day with a sample made from *their* app.
- Indie Hackers and the RevenueCat Sub Club community.
- Product Hunt launches from the last 30 days: every one of them needs launch clips.

**30-day plan.**
1. Days 1-3: build the pipeline. Script generator (Claude), AI actor tool (Arcads or
   MakeUGC trial), caption/format step, export presets for 9:16. Make 10 sample ads for
   3 real apps you pick from Product Hunt.
2. Days 4-10: outbound. 20 personalized DMs/day with a free sample. Post the samples as
   "before/after" on X and TikTok. Sell the $297 Sprint first; upsell to Starter.
3. Days 11-30: fulfil, collect testimonials, raise prices after 5 customers. Post every
   day: "we made 40 ads for an ADHD app in 2 hours, here's what won."

**Risks.** AI-actor platforms change pricing and ToS; keep two vendors. Buyers churn if
variants don't convert, so include a weekly "what won" report to make it sticky.

---

## Lead 2 (build this app): ADHD "AI body double" iOS app

**What it is.** A real-time voice companion that sits with you while you do the thing:
dishes, laundry, email, taxes. You say "I need to clean the kitchen," it breaks it into
5-minute chunks, talks you through them out loud, checks in, and celebrates. Think
Focusmate without scheduling a stranger, or dubbii without pre-recorded videos.

**Why this niche.**
- Body doubling is an established ADHD coping tool with paying competitors:
  Focusmate $6.99/mo (1:1 human video, must be scheduled), Flow Club $19-$25/mo,
  FLOWN $19-$25/mo, dubbii (subscription, pre-recorded videos).
- Competitor gaps from App Store reviews: dubbii users complain it is "just two social
  media creators making how-to videos," that pricing is unclear at signup, and that live
  sessions fail to join. Focusmate and Flow Club require booking and a camera. Nobody
  offers an always-on, on-demand, voice-first double.
- Distribution exists already: #ADHDTikTok crossed ~2 billion views, 3.9M posts on #ADHD,
  top videos average 10M views, and 77% of college students have seen ADHD content.
- Health & Fitness is the highest trial-to-paid category on iOS (35%); hard paywalls
  convert at a median 10.7% day-35 vs 2.1% for freemium.
- No Apple entitlement gate. Voice, notifications, Live Activities only.

**Pricing.** Hard paywall, 3-day trial, $9.99/mo or $49.99/yr (push annual: annual
subscribers retain 44% at 12 months vs 17% for monthly). $10K MRR is roughly 1,000
monthly subscribers or a mix of ~1,500 monthly/annual.

**MVP scope (2-3 weeks).**
- Onboarding: "what's the task?" then AI splits into steps.
- Voice session: TTS + STT loop with a warm persona, 5-minute timers, check-ins.
- Live Activity on lock screen showing current step and timer.
- Streaks and a "done pile" the user can share as a vertical video (the growth loop).
- Stack: SwiftUI, RevenueCat, a realtime voice API, Claude for task breakdown.

**Content engine (uses Lead 1's pipeline).** Daily posts: "POV: my body double talks me
through the dishes" screen recordings with real audio, ADHD-humor hooks, AI persona
accounts posting the same clips in 3 voices. One viral hit is the goal; the pipeline
makes 30 shots at it per month cheap.

**Risks.** Voice API cost per session: cap free sessions, meter paid ones. Avoid medical
claims; position as a productivity companion, not treatment.

---

## Lead 3 (apply on day 1, ship second): Anti-brain-rot screen-time app for Gen Z

**What it is.** A social app blocker where you and a friend put streaks or small stakes
on the line: if you open TikTok during a focus block, your friend gets notified and you
lose the streak. Opal for people who won't pay $19.99/mo and want a friend, not a wall.

**Why the demand is the strongest of the three.**
- Opal: $17.1M ARR, 1M DAU, two-thirds of users are high school and college students,
  and the switch to freemium is what unlocked that growth.
- one sec charges $2.99/mo; ScreenZen is free; Opal is $19.99/mo. The middle ($4.99/mo
  with a social hook) is open.
- 46% of Gen Z say they are actively trying to reduce screen time. #dopaminedetox and
  "brain rot" content is a top TikTok genre in 2026, and marketing an anti-TikTok app on
  TikTok is a proven ironic hook.

**Why it's ranked third.** Apple's FamilyControls entitlement is required to block apps,
and in 2026 indie developers report 10-14+ day waits with no response, plus a second
wait for the Shield extension. Apple support has admitted a backlog. That gate alone
makes this a 6-8 week project, not 30 days. Submit the entitlement request on day 1 so
it is approved by the time Lead 2 ships.

**Pricing.** Freemium (that is what worked for Opal): free for one blocked app and one
friend, $4.99/mo or $29.99/yr for unlimited, stakes, and stats.

---

## Rejected on purpose

- **Sleepmaxxing app.** Trend is real (100M+ TikTok posts, searches up 300%) but the
  category is hardware-led (Oura filed for IPO at $11B), and the 2026 narrative has
  flipped to "from sleepmaxxing to simplicity" with clinician backlash. Late and crowded.
- **B2B tool mined from G2/Capterra.** Almost everything on G2 scores above 4.0 because
  vendors fund review campaigns, so the signal is weak. B2B sales cycles are also too slow
  for the cash goal. Use these sites for research only (below).

## How to use the sources you listed

- **Reddit.** The best pain-signal source for B2C. Search r/ADHD, r/getdisciplined,
  r/nosurf, r/productivity for "app", "wish", "alternative to", "cancelled". Sort by new.
  Reddit blocks most scrapers; use its own JSON endpoints (append `.json` to a thread
  URL) with a proper User-Agent, rate-limited.
- **Firecrawl.** Free tier is 500 credits, 10 scrapes/min, and it explicitly blocks
  TikTok, Instagram, and YouTube. Fine for App Store review pages, competitor sites,
  and Product Hunt. Not a Reddit or TikTok tool.
- **App Store reviews.** The richest gap finder for Leads 2 and 3. Pull 1-2 star reviews
  of every competitor named above; each complaint is a hook for a TikTok post.
- **Capterra / G2.** B2B only, and G2 now owns Capterra. Useful if you ever sell Lead 1
  to agencies. Capterra's incentivized-review labeling makes its 1-star reviews more
  trustworthy than G2's.
- **Consumer Reports.** Not useful here; it covers physical goods and its own app is
  poorly reviewed. Skip.

## Sequence

| Week | Lead 1 (service) | Lead 2 (ADHD app) | Lead 3 (screen time) |
|---|---|---|---|
| 1 | Build pipeline, 30 samples, 100 DMs | Spec + paywall design | Submit FamilyControls request |
| 2 | First 3 paying Sprint customers | Build voice loop + Live Activity | Wait |
| 3 | Upsell to Starter, daily posts | TestFlight, 20 ADHD beta users | Wait |
| 4 | 8-10 customers ($4-6K MRR) | App Store submit, content blitz | Start build if approved |

If Lead 1 does not have 3 paying customers by day 14, the offer or the targeting is
wrong; fix that before touching the app.
