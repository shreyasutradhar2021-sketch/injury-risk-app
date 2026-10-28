# Club Medical & Load-Management Protocol (v1)

Educational prototype written by Team 8 for a fictional ISL club. Not medical advice.
Sources the clauses are based on: Gabbett (2016) on acute:chronic workload ratios,
Milewski et al. (2014) on sleep and injury, Hägglund et al. (2006) on previous injury,
and the Ardern et al. (2016) consensus statement on return to sport.

The app reads this file fresh every time. Each clause has an `Applies when:` line that
the app checks against the player's numbers to retrieve the clause. Changing a number
here (for example `acwr > 1.5`) changes which clauses are retrieved, with no code change.

Settings:
- borderline_band = 0.10
- min_similarity = 0.10
- conflict_pairs = C-3:C-7, C-2:C-5

## C-1 Purpose and authority
Applies when: always
This tool supports the medical team; it does not make decisions. No change to a player's
training status takes effect until the club doctor or the head physiotherapist signs it off.
Every briefing, sign-off and override is recorded in the audit log.

## C-2 Low risk
Applies when: risk_level == Low
The player continues the planned training week. Routine weekly monitoring continues and
the next weekly data entry is reviewed as normal.

## C-3 High risk
Applies when: risk_level == High
The medical team reviews the player within 24 hours. The medical team may reduce next
week's planned distance by 20 to 30 percent and high-intensity distance by up to 50 percent,
and may add one extra recovery day. The head coach is informed only of the signed-off outcome.

## C-4 Borderline score
Applies when: borderline == yes
When the risk score is within 10 percent of the cut-off, no automatic change is made.
A physiotherapist completes a short wellness check (soreness, sleep, mood) before the next
session, and the medical team decides whether to treat the player as Low or High risk.

## C-5 Workload spike
Applies when: acwr > 1.5
An acute:chronic workload ratio above 1.5 is a workload spike. Next week's total distance
should not increase by more than 10 percent, and very hard sessions should be spread out
with at least 48 hours between them.

## C-6 Workload caution zone
Applies when: acwr >= 1.3 and acwr <= 1.5
An acute:chronic workload ratio between 1.3 and 1.5 is a caution zone. Staff keep next week's
distance at the current level and monitor wellness scores daily.

## C-7 Under-loading
Applies when: acwr < 0.8
An acute:chronic workload ratio below 0.8 means the player is under-loaded, which also raises
injury risk when match load returns. Load should be rebuilt gradually by 10 to 15 percent per
week. Load should not be reduced further unless the medical team decides otherwise.

## C-8 Poor sleep
Applies when: sleep_hours < 7
Average sleep below 7 hours per night is reported to the medical team. The player receives
sleep-hygiene guidance, and high-intensity sessions are avoided on the morning after a
night of poor sleep where the schedule allows.

## C-9 Previous injury
Applies when: prior_injury_12m == 1
A player with a time-loss injury in the past 12 months is monitored more closely. The
physiotherapist checks that the injury-prevention and strength programme from rehabilitation
is still being followed.

## C-10 High perceived exertion
Applies when: exertion >= 6
An average perceived exertion of 6 or more out of 10 suggests accumulated fatigue. Staff ask
the player about fatigue and soreness and consider replacing one hard session with a lighter one.

## C-11 Rest days
Applies when: rest_days == 0
Every player has at least one full rest day per week. If the player had no rest day this week,
one is scheduled in the coming week.

## C-12 High-intensity share
Applies when: hi_share > 0.15
When more than 15 percent of weekly distance is at high intensity or sprint speed, the
following week includes at least one low-intensity recovery session.

## C-13 Return to training after injury
Applies when: question_only
Return after an injury follows four stages: rehabilitation, modified training, full training,
then match play. Each stage needs the club doctor's approval. The model's risk score is never
used on its own to move a player between stages.

## C-14 Congested fixtures
Applies when: question_only
In a week with two matches, planned training distance is reduced to allow recovery between
matches. Rotation and selection decisions belong to the head coach, informed by the signed-off
medical status.

## C-15 Data quality
Applies when: question_only
If weekly data is missing or outside plausible ranges, no risk score is produced. Staff
re-enter the data; values are never guessed or filled in by the system.

## C-16 Privacy and communication
Applies when: always
Player health data is personal data under India's Digital Personal Data Protection Act 2023.
The head coach sees only the signed-off status and briefing, not raw wellness scores. Player
health information is never shared with media, agents or other players.

## C-17 Conflicting clauses
Applies when: conflict == yes
When two clauses point in different directions, the briefing states the conflict and names
both clauses. Until the club doctor decides, the more cautious option applies: no increase in
load and a medical review within 24 hours.
