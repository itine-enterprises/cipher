# The Mindbody Exit Offer, built on the Isenberg / Ganim audit model

Source: The Startup Ideas Podcast, Greg Isenberg with Corey Ganim, "The $1,000/hour Solo AI
business (Full Course)". Corey sells a $999 "AI Tools Assessment" to small businesses and
converts about half into implementation and retainer work. Below is his system, quoted,
and then the same system pointed at one-location studios stuck on Mindbody.

## Corey's system, as stated on the episode

**The offer.** A $999, 45-minute assessment for owners of 2-20 person businesses doing
$500K-$5M. Output is 3-7 off-the-shelf tools. Guarantee: "if we can't find at least five
hours per week in opportunity where AI can help, then we're going to give you 100% of your
money back." He has never refunded.

**Phase 1, discovery call (30-45 min, recorded with Fathom/Otter).** "We're just asking
questions. We're pulling out problems. We're not pitching anything." The questions:
- Walk me through your day yesterday.
- What tasks do you dread doing?
- Where does work pile up?
- What have you tried to automate before that failed?
- If you could wave a magic wand and delete one process, which one?

**Phase 2, AI analysis.** Transcript goes into Claude: "attached is the transcript of our
conversation, research off-the-shelf tools that fix these pain points." "Claude is going to
catch patterns you might have missed." Then a human sanity pass (swap Salesforce for a
small CRM for a 4-person business). By the 5th or 6th run it is a saved skill.

**Phase 3, the report (Gamma, now Claude Design; template at audittemplate.ai).** Nine
slides: title, executive summary (1-2 pain points, hours reclaimed, primary lever),
effort-vs-impact matrix, quick wins, recommended solutions (tool, pain point, cost, setup
time, weekly savings), a four-day quick-start plan, "what comes after quick wins" (the
upsell tease), financial impact (hours x hourly rate minus tool cost, "always four figures,
sometimes five"), next steps.

**Phase 4, review call (30 min, screen-share).** Walk each recommendation. Three closing
questions: "Which of these is most urgent for you?" "Do you want to do this yourself, or
would you like my help implementing?" "What's your timeline?" roughly half buy implementation (Corey's self-reported figure; episode aired July 13, 2026).
The $999 is credited toward it ("instead of 5K, it's 4K").

**Upsell menu.** Process redesign $3,000-3,500. One Zapier/Make build ~$1,500. Custom
workflows $3,000+. Full implementation $8,000+. "AI Concierge" retainer $1,200 rising to
$2,000/mo for two 45-min calls plus Voxer; 5 of the first 6 prospects took it, $8K MRR in
10 days. Four assessments plus four retainers a month is $10K.

**Getting clients with no money.** Day 1: text 20 people in your warm network. Weeks 1-2:
free 20-minute mini-audits. Then $200, then $999 after 3-5 paid. Door-knock 30 local
businesses (5 meetings, 2 clients). LinkedIn first message is a probe, not a pitch.
Partnerships with accountants and agencies for referral fees. Weekly office hours at a
coworking space. Post wins for 90 days.

**Why Greg says the audit works.** "A confused mind doesn't buy, doesn't implement, doesn't
upsell," so the output has to be "extremely stupid, simple." "Your client is literally
paying you to uncover opportunities for them to pay you more." "It's a lot easier to pitch
a $5,000 package to somebody who just paid you $1,000." Niche by geography or vertical:
"the AI guy in Charlotte" or "AI for financial services."

## The same system aimed at Mindbody studios

Corey's model is generic (any small business, any pain). Ours is narrower and therefore
easier: one vertical, one villain, one outcome. Every studio owner on r/mindbody already
knows the pain; the audit's job is to put a number on it and a date on the exit.

**Positioning.** "The Mindbody exit guy." Niche by vertical (studio and gym: Pilates,
yoga, barre, dance, martial arts, small gyms), not geography, because the customers are
already gathered in five subreddits and one Capterra review page.

### The offer: Mindbody Exit Audit, $199 (raise to $499 after five paid)

Corey charges $999 because his ICP does $500K-$5M. A single-location studio does
$150K-$600K and is currently annoyed about a $700/mo bill, so start lower and let the
migration carry the ticket. His ramp applies exactly: first two free, then $199, then $499.

**Guarantee.** "If the audit does not find at least $1,500 a year in savings or a
contract exit you did not know you had, full refund." The Capterra and Reddit data says
Capterra and BBB reviews consistently describe year-over-year increases and features moved
into pricier tiers, and Mindbody's terms auto-renew for 12 to 36 months, so this guarantee is safe.

### Phase 1: discovery call, 30 min, recorded

Same rule: pull problems, pitch nothing. The studio version of Corey's questions:
- Walk me through what happens in Mindbody from a new client's first booking to their
  autopay renewal.
- What do you dread doing in it every week?
- Where do things pile up? (Front desk check-in, package credits, failed payments, reports
  you rebuild in Google Sheets.)
- What did you try to fix or leave before, and what stopped you?
- Magic wand: which Mindbody process would you delete?
- The three exit-specific questions: When does your contract renew and what notice does it
  require? How many clients are on autopay with a stored card? Do you have a branded app or
  website widget tied to Mindbody?
Ask them to send the contract, the last three invoices, and a screenshot of their pricing
options before the call.

### The stored-card reality, verified

Vagaro's and WellnessLiving's own help articles say they do not import stored cards
(WellnessLiving: "cannot be imported due to privacy and security regulations"; Vagaro:
"Contact Mindbody"). TeamUp imports cards only from a Stripe or GoCardless account the studio
already owns. Mindbody Payments runs on Stripe, but Mindbody owns the account, so Stripe's
processor-to-processor export can only be initiated by Mindbody, through its paid encrypted
export (reported at about $500). Three cases, decided in the audit: (A) Mindbody paid export
to a destination that accepts it, (B) TeamUp import from the studio's own Stripe, (C) a re-card
campaign during the parallel week: email and text sequence, tablet at check-in, first charge
confirmed before Mindbody autopays are cancelled. Case C is the common one and is most of the
migration's value.

### Phase 2: analysis in Claude, saved as a skill after the third one

Feed the transcript, the invoices, and the contract. The prompt is Corey's with the
target changed: "find the true monthly cost including add-ons and processing on their
actual volume, find the renewal window and exit terms, list everything locked in (stored
cards, package credits, waivers, app, widget), and compare against Vagaro, WellnessLiving,
Vibefam, Momence, and Glofox at this studio's size and type." Human sanity pass: match
platform to studio type (Pilates single-location vs. strength gym with programming).

### Phase 3: the report, 2 pages, same skeleton as Corey's nine slides

1. Title: studio, date, current plan, renewal date.
2. Executive summary: what you pay now, what you would pay after, what leaving costs.
3. Lock-in matrix (effort vs. risk): stored cards, package credits, class history,
   waivers, app, widget. Each with the known fix.
4. Quick wins: things they can do inside Mindbody this week regardless (record the
   downgrade in writing, clear the marketplace fee setting, pull the client export now).
5. Recommended platform: one, with price at their volume and why.
6. Four-week exit plan, dated: notice, parallel run, cutover, card-data clearance.
7. What comes after: the migration package (the upsell tease).
8. Financial impact: annual savings minus migration fee, payback in months.
9. Next steps and the review call.

Build it once in Claude Design from audittemplate.ai and reuse it.

### Phase 4: review call, 30 min, Corey's three questions unchanged

"Which of these is most urgent?" "Do you want to do the migration yourself, or would you
like my help?" "What's your timeline?" The $199 is credited against the migration.

### The upsell menu, priced for studios

| Package | Price | Corey's equivalent |
|---|---|---|
| Full Migration | $1,200-1,800 | Full implementation |
| Stored-card handling only: Mindbody export request, or re-card campaign | $500 | One Zapier build |
| Reporting rebuild (the Google Sheets dashboard they already keep) | $400 | Process redesign, small |
| Studio Ops retainer | $149-299/mo, one call a month plus text access | AI Concierge |

Corey's math, at our prices: six audits a month at $199 is $1,200; three of them convert
to migration at $1,500 is $4,500; every migration that takes the retainer adds $149-299.
Month three with ten retainers on the books is $7K-9K before new audits. Vendor referral
fees from the target platforms sit on top.

### Getting the first ten, Corey's list translated

- **Warm network, day 1:** anyone who teaches, trains, or owns a studio. Free 20-minute
  mini-audit: "tell me your plan and I'll tell you what you're overpaying."
- **The 54 Reddit switching posts:** reply with one specific number from their post, then
  offer the free mini-audit. First message is a probe, never a pitch.
- **Capterra 1-star reviewers:** name, role, studio type are public. Email the same probe.
- **Door-knock equivalent:** walk into ten local studios with a Mindbody booking widget on
  their site. Five conversations, two audits.
- **Partnerships:** the vendors (Vagaro, WellnessLiving, Vibefam, Momence, Glofox) are the
  "accountants and agencies" here. Ask for referral fees; they are already astroturfing
  the same threads.
- **Post wins for 90 days:** "Studio X was paying $812/mo. After the exit: $189." One per
  migration, on X, r/mindbody where allowed, and TikTok.

### What is different from Corey, on purpose

- He sells hours saved; we sell dollars saved and a date. Studios do not need convincing
  that there is a problem, so the report is a decision, not a discovery.
- He stays generic; we own one villain. That gives the content a hook ("Mindbody exit
  guy") and makes the skill in Phase 2 converge after three audits instead of six.
- His retainer is coaching; ours is operational (reporting, renewals, failed payments),
  which studios already pay Mindbody for and hate.
