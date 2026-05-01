# Debt Collection Call Workflow (English)

## Flowchart (high level)

```mermaid
flowchart TD
  A([Start call]) --> S1[Stage 1: Verify right party\n"Hi, is this Mr. {name}?"]

  S1 -->|DNC request| END_DNC([End: DNC])
  S1 -->|Death reported| END_DEATH([End: Death])
  S1 -->|Busy / can’t talk| END_BUSY([End: schedule callback time])

  S1 -->|Wrong party| WPQ[Ask: "Sorry do you know Mr. {name}?"]
  WPQ -->|No| END_WRONG([End: "Thanks have a good day."])
  WPQ -->|Yes| WPN[Collect 10-digit mobile\nConfirm once then end]

  S1 -->|Right party confirmed| ID[Stage 1: ID last4 yes/no\n"To verify, are the last 4 digits {id_last4_words}?"]
  ID -->|Mismatch/refusal| END_WRONG
  ID -->|Match| S2[Stage 2: QA + debt intro + reason\n"This call may be recorded for quality."\n"You have an overdue balance of {amount} SAR what’s preventing you from paying it?"]

  S2 -->|Says paid| END_PAID([End: “record will update soon”])
  S2 -->|Denial or reason captured| S3[Stage 3: Negotiation ladder]

  S3 --> C[Say once: 7-day SLA + SIMAH/procedures]
  C --> L1[Attempt 1: full today/tomorrow]
  L1 -->|Yes| COMMIT[Commitment captured\n(amount + exact date)]
  L1 -->|No| L2[Attempt 2: half exception]
  L2 -->|Yes| COMMIT
  L2 -->|No| L3[Attempt 3: customer-named instalments]

  L3 -->|Plan agreed| COMMIT
  L3 -->|Still vague timing| RESCH[Reschedule callback time]
  L3 -->|Refusal| REFUSE([End: refusal])
  L3 -->|Dispute/denial| DISPUTE([End: dispute])

  COMMIT --> S4[Stage 4: recap + payment methods + close\n"To confirm, this is the plan we agreed on correct?"]
  RESCH --> S4
  S4 --> END_OK([End])
```

---

## Stage model (hard boundaries)

The call is a strict sequence:

1. **Stage 1** → right-party verification + ID-last4 yes/no check  
2. **Stage 2** → QA recording disclosure + overdue invoice intro + reason question  
3. **Stage 3** → negotiation ladder (commitment/reschedule/refusal/dispute)  
4. **Stage 4** → recap plan + confirm + close  

Stage entry lines are intentionally **fixed** to keep boundaries consistent and avoid drifting into unrelated talk.

---

## Stage 1   Right-party verification + ID last4

### Goal

- Confirm the callee is the intended person (right party), then perform a final yes/no ID-last4 check.
- **No debt or account details are allowed** in Stage 1.

### Deterministic stage entry

Use exactly one of these (one question):

- `"Hi, is this Mr. {name}?"`
- (If the customer greeted first) `"Hi yes, is this Mr. {name}?"`

### Challenge handling

If the callee asks a challenge question, answer in **one short sentence**, then immediately return to the pending verification question.

Canonical replies:

- **Who are you?**: `"This is Nora from Tawafuq, calling on behalf of Mobily."`
- **Why calling?**: `"I’m calling regarding your Mobily account."`
- **Where did you get my number?**: `"Your number is from Mobily’s registered records you can verify via 1100."`

### Final ID verification step

After the right party is confirmed, Nora asks the last4 as a strict yes/no question:

- Step 1 (yes/no only): `"Just to verify, are the last 4 digits of your ID / residency {id_last4_words}?"`
- Step 2 (retry once only): `"Let me repeat that: {id_last4_words}. Is that correct?"`

### Outcomes (Stage 1)

Business outcomes:

- **Confirmed** → proceed to Stage 2
- **Denied / wrong party** → polite close
- **Busy / can’t talk** → schedule a callback time
- **DNC / death** → end immediately (override)

Canonical closing lines:

- **Wrong party / denied**: `"Thanks have a good day."`
- **Busy**: `"No problem what time works best for me to call you back?"`

### Wrong-party handoff (if they know the account holder)

If the callee is **not** the named person but says they know him:

- Ask: `"Sorry do you know Mr. {name}?"`
- If YES: `"Can you share his 10-digit mobile number so we can reach him?"`
- Repeat once: `"Let me repeat the number to confirm: {number}. Is that correct?"`
- Close: `"Thanks for your help have a good day."`

---

## Stage 2   Disclosure + debt intro + reason capture

### Deterministic stage entry (script)

1. QA recording disclosure (short):

- `"Thanks, Mr. {name}. This call may be recorded for quality."`

1. Debt intro + reason question (verbatim intent):

- `"I’m calling about your Mobily account there’s an overdue balance of {amount} SAR that hasn’t been paid yet. What’s the reason for the delay?"`

If the customer answers with “نعم/طيب/تمام/اوكي” without giving a reason, re-ask once:

- `"I mean what’s preventing you from paying it?"`

If the customer says they already paid (سددت/دفعت/تم الدفع):

- `"Thank you. Noted we’ll update your record as soon as possible. Thanks."`
- Then end the call.

### Intended responsibility

Stage 2’s job is **only** to capture a reason (or detect denial) and then transition to Stage 3.

### Outcomes (Stage 2)

Business outcomes:

- **Reason captured** → proceed to Stage 3
- **Immediate denial** → proceed to Stage 3 (but treat as dispute/denial path)
- **DNC / death** → end immediately (override)

---

## Stage 3   Negotiation & commitment capture (installments)

### Deterministic stage entry

Stage 3 begins *after Stage 2 captured the reason*.

1. **Consequences line** (once per call, non-threatening, procedural framing):

- `"Just so you’re aware: we need this resolved within 7 days. Otherwise it can affect your credit record and may be reported to SIMAH, and it can be escalated per policy."`

1. **Attempt 1** (first ladder question, full payment):

- `"Understood. Can you pay the full amount today or tomorrow?"`

### Negotiation ladder (how installments work)

Stage 3 uses a ladder to converge on a concrete plan (amount + date):

- **Attempt 1 (full)**:
  - `"Can you pay the full amount today or tomorrow?"`
- **Attempt 2 (half exception)** (if full is declined):
  - `"As an exception, can you pay half today/within two days, and the remaining balance on a date you choose?"`
- **Attempt 3 (customer-named tranche / instalments)** (if half is declined too):
  - `"What’s the minimum amount you can commit to, and on what exact date?"`
  - Then follow with: `"And when can you pay the remaining balance?"`

### Vague timing handling (timezone-aware)

If callee says “آخر الشهر / هذا الأسبوع / عند نزول الراتب”:

- Ask **one** clarifying question to pin an exact day/date (may suggest month-end day).
- **Never assume** a date from vague phrases (do not convert “salary” to “end of month” unless confirmed).
- If still vague after one clarification: switch to callback scheduling and record reschedule.

Canonical clarification:

- `"Which exact day do you mean say the 30th, for example?"`

Canonical reschedule (if still vague):

- `"Okay let’s set a time for me to call you back so we can confirm the exact date. What time works for you?"`

### Outcomes (Stage 3)

- **Commitment**: clear plan agreed (amount(s) + exact date(s)).
- **Reschedule**: callback time agreed to confirm exact plan/date.
- **Refusal**: callee refuses any commitment after offers.
- **Dispute/denial**: callee denies owing or disputes the debt.
- **DNC / death**: end immediately (override).

---

## Stage 4   Recap & close

### Trigger

Stage 4 runs only after Stage 3 returns **commitment** or **reschedule**.

### Deterministic behavior

- Restate the agreed plan once (amount + date).
- Add a short line that Nora will contact the callee on the agreed date to confirm.
- End with one confirmation question: “مضبوط؟”

Canonical template:

- `"Great just to confirm, this is the plan we agreed on: {plan_summary}. You can pay via SADAD using the account number, through the Mobily app/website using your national ID, or by visiting the nearest Mobily branch. I’ll call you on the agreed date to confirm. Is that correct?"`

