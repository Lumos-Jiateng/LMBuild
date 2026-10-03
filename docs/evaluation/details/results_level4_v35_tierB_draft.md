# LMBuild Level 4 (Realization), spec v3.5, Tier B, pass 3: DRAFT pending the owner's approval

Generated on 2026-09-22 from `results/v2/*/eval_v35/` by `python -m ppbench.v2.spec_v35 build --scope B`.

- **Setting:** Tier B, `name_only+image`, one 3-round trajectory, final design, seed 0.
- **Systems:** 19 LLMs. Scores are ×100.
- **Not yet applied:** Tier A, Tier C, the ablations and the baselines.
- **Owner's rule:** a design that declares no sequence, material or roles scores 0 on that dimension.

**Notation.** A design has $n$ parts. Its declared order is $\pi = (\pi_1,\dots,\pi_m)$ with $m \le n$; parts it omits are never placed. The state after step $k$ is $X_k = \{\pi_1,\dots,\pi_k\}$.

---

## P.1 Sequence: can a robot build it, and how safe is every intermediate state?

$$\text{P.1} = \tfrac12\,E + \tfrac12\,S$$

**Executable prefix.**

$$E = \frac{k^\*}{n}, \qquad k^\* = \min\{k : \text{step }k\text{ is not executable}\} - 1 \quad (k^\* = m \text{ if every step is executable}).$$

**State robustness.** Every step is checked, including the steps after the first failure.

$$S = \frac{1}{n}\sum_{k=1}^{m} s_k, \qquad
s_k = \begin{cases} 0 & \text{step }k\text{ not executable}\\[2pt]
\min_{B \in \mathcal{B}(X_k)} \min\!\Big(1, \dfrac{\theta_B}{\theta^\*}\Big) & \text{otherwise} \end{cases}$$

**Symbols.**
- $\mathcal{B}(X_k)$ are the bodies of state $X_k$. Connected parts (touching, or joined by a declared joint) form one rigid solid.
- $\theta_B$ is body $B$'s critical tilt angle on the floor.
- $\theta^\*$ is the task's 1.3 target tilt: the reference's tilt clipped to $[5^\circ, 30^\circ]$, or $30^\circ$ when the task has no reference.
- A state that barely stands is therefore *vulnerable* and earns less.

**Step $k$ is executable** if and only if both of these hold:
1. **Insertion.** Part $\pi_k$ has a straight collision-free approach, along its declared direction or one of the axes, past $X_{k-1}$. The approach never comes up through the floor.
2. **Support.** Every body of $X_k$ touches the floor, and every body has $\theta_B > 0$.

The build is never re-oriented. For tasks that are not free-standing, $\theta$ is not required.

**Reported, not scored:** the share of valid states, a greedy order search (the best reachable $E$), and the strict "one body at every step" variant.

---

## P.2 Material: is every part made of a material that fits its function?

$$\text{P.2} = \frac{1}{|I|}\sum_{i \in I} a_i, \qquad
a_i = \begin{cases} 1 & c_i \in A_i\\ 0.5 & \text{fam}(c_i) \in \text{fam}(A_i)\\ 0 & \text{otherwise, or no library material declared} \end{cases}$$

**Symbols.**
- $c_i$ is the declared material class of part $i$. There are 9 classes: heavy metal, light metal, plastic, wood, glass, ceramic, rubber, foam, textile.
- The families are metal, polymer, soft, wood, glass and ceramic.
- $I$ is the set of parts with a non-empty accepted set $A_i$.

**The accepted set $A_i$** uses evidence first, and falls back to the function judgement only when there is no evidence:

$$A_i = \begin{cases} W(r_i) \,\cup\, P(r_i) \,\cup\, R(i) & \text{if non-empty}\\ J(\text{name}_i) & \text{otherwise} \end{cases}$$

**Symbols.**
- $r_i$ is the part's canonical role.
- $W(r)$ is the wiki evidence for the role, from two sources:
  - the task's verified material claims;
  - materials read by Qwen3.5-27B from the cited Wikipedia pages at their pinned revisions. An answer is kept only if all three checks pass:
    - the quote is verbatim in the page;
    - every material it names is a material word in that quote;
    - the quote does not date itself (e.g. "century", "originally", or a year before 1950).

    177 of 552 major parts were kept.
- $P(r)$ is the set of classes held by at least 10% of that role's parts across the task's Artiverse category, counting interior and exterior annotations.
- $R(i)$ is the set of classes of the reference parts in the same place as part $i$. A reference part counts when it covers at least 25% of part $i$ on the scale-normalised grid.
- $J(\text{name})$ is the class set from a function judgement. Qwen3.5-27B is given only the object and the part's name, never the design's material. 11,262 part names were judged, and 28 got no class.

**Coverage in Tier B:** 4,774 parts are scored on evidence, 3,119 on judgement, and 7 have no basis.

---

## P.3 Operability: does every functional chain the wiki names actually work?

$$\text{P.3} = \frac{1}{|C|}\sum_{c \in C} w_c, \qquad w_c = \min_{\ell \in L_c} \ell \quad \text{(a chain is as strong as its weakest link)}$$

The capabilities $C$ all come from the task's cited claims (sheet `op-v2`).

**F: one capability per functional subsystem.** Its links $L_c$ are:
- **presence:** $\mathbb 1[\text{each named part is present}]$. Internal parts count too, and a part named "X ⟨WordNet part of X⟩" counts as X.
- **path:** a path between the parts of each named power or motion connection, of at most 4 hops. A motion path must pass through a moving joint *and end on a part that can move*.
- **control:** the control is within reach, and it drives the moving part through a path of at most 6 hops.

**M: one capability per claimed motion K.** It is simulated in MuJoCo; the rules are listed below.

**H: one capability per human-use check** from the authored layer, e.g. "can be stopped" = brakes present ∧ brakes within 0.2 m of the wheels.

**P: required parts outside every subsystem.** All of them must be present.

**Reported, not scored:** $\min_c w_c$ ("fully operable"), and the share of capabilities with $w_c \ge 0.5$.

### M: the simulation (`ppbench/v2/sim_v35.py`, MuJoCo 3.14.0)

**Motion score.**

$$w_M = \text{achieved} \times \mathbb 1[\text{tilt}_{\max} < 15^\circ]$$

**Achieved** depends on the kind of motion:
- **Roll.** Applies when the moving part is a wheel, caster, roller or track. The object is pushed horizontally with $0.1\,mg$ for $1.5$ s, with every hinge between a floor-touching body and the base released.

  $$\text{achieved} = \min\!\Big(1, \frac{d}{0.25\, d_0}\Big), \qquad d_0 = \tfrac12 (0.1g)\,t^2 \ \text{(frictionless distance)}$$

- **Actuate.** Applies to any other moving part. A servo drives the design joint that lies between the moving part and its claimed reference part, and has the claimed type, through the declared range (else ±90°, a full turn, or ±half the part's length), trying both directions.

  $$\text{achieved} = \frac{|\Delta q|}{|q_{\text{target}}|}$$

**Model rules.**
- Connected parts are one rigid body, except that contact never welds across a declared moving joint.
- Masses use the 1.3 model.
- Shapes are convex hulls. A round floor-touching body centred on a horizontal hinge collides as its cylinder of revolution.
- A door collides with its own cabinet.
- Pairs that already interpenetrate in the design are exempt.
- Objects that are not free-standing are mounted.
- Validation: the sim and static 1.3 agree on standing/falling for 91% of designs. Golden scenes: a free-wheel cart rolls, a fixed-wheel cart does not, a lid opens, and a blocked flap is stopped.

---

## Results: Tier B

| system | **P.1** | E | S | **P.2** | parts on evidence % | **P.3** | F subsystems | M motions (sim) | H human use | fully operable | N |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Closed-source APIs** | | | | | | | | | | | |
| GPT-6 Astra | **64.9** | 55.2 | 74.5 | **78.1** | 65 | **39.3** | 24.8 | 39.0 | 69.9 | 4.8 | 40 |
| GPT-5.6 Sol | **64.2** | 54.5 | 73.9 | **84.3** | 59 | **36.6** | 21.0 | 35.4 | 64.3 | 4.8 | 40 |
| Claude Opus 5 | **52.7** | 39.3 | 66.1 | **79.7** | 61 | **40.6** | 24.6 | 41.2 | 66.1 | 4.8 | 40 |
| Claude Fable 5.1 | **65.1** | 58.8 | 71.4 | **76.2** | 62 | **37.2** | 20.7 | 37.8 | 67.5 | 7.3 | 40 |
| Claude Sonnet 5 | **72.3** | 68.7 | 76.0 | **80.9** | 51 | **31.3** | 19.5 | 29.4 | 66.0 | 5.8 | 50 |
| Claude Haiku 4.5 | **56.1** | 50.6 | 61.6 | **81.1** | 53 | **20.5** | 13.4 | 15.5 | 46.8 | 2.0 | 50 |
| **Open-source models** | | | | | | | | | | | |
| Qwen3.5-35B-A3B | **59.9** | 57.9 | 61.9 | **78.0** | 59 | **23.5** | 10.8 | 17.2 | 54.3 | 3.8 | 50 |
| gpt-oss-120B | **58.1** | 58.0 | 58.3 | **80.1** | 56 | **22.5** | 11.0 | 15.6 | 55.7 | 3.8 | 50 |
| Qwen3.5-27B | **74.1** | 75.4 | 72.7 | **79.5** | 62 | **22.1** | 11.8 | 12.0 | 52.3 | 4.0 | 50 |
| Gemma-4-31B-IT | **54.5** | 52.5 | 56.5 | **77.3** | 61 | **19.4** | 10.1 | 14.5 | 45.9 | 1.9 | 50 |
| Qwen3-VL-32B | **32.6** | 32.4 | 32.9 | **62.2** | 68 | **16.2** | 9.5 | 12.1 | 35.7 | 2.0 | 50 |
| Ministral-3-8B | **28.9** | 25.2 | 32.7 | **78.9** | 55 | **15.1** | 9.8 | 7.3 | 39.1 | 2.0 | 50 |
| Qwen3-4B-Instruct | **37.6** | 33.9 | 41.4 | **74.1** | 60 | **14.3** | 8.6 | 9.6 | 29.1 | 0.5 | 50 |
| Qwen3-VL-30B-A3B | **32.2** | 35.2 | 29.1 | **69.6** | 70 | **13.3** | 10.8 | 4.2 | 29.1 | 2.0 | 50 |
| InternVL3.5-38B | **21.3** | 20.3 | 22.2 | **68.2** | 60 | **13.1** | 8.3 | 10.5 | 24.5 | 0.0 | 50 |
| Qwen3-VL-8B | **38.0** | 35.4 | 40.7 | **62.6** | 68 | **12.9** | 6.8 | 8.8 | 17.7 | 2.0 | 50 |
| ERNIE-4.5-VL-28B-A3B | **11.0** | 10.9 | 11.1 | **53.9** | 70 | **12.8** | 7.1 | 5.7 | 27.2 | 2.0 | 50 |
| MiniCPM-V-4.5 | **10.0** | 9.0 | 11.0 | **65.3** | 57 | **9.6** | 7.0 | 1.6 | 27.2 | 0.0 | 50 |
| InternVL3.5-8B | **2.9** | 2.9 | 2.8 | **54.2** | 65 | **5.8** | 2.6 | 4.2 | 10.7 | 0.0 | 50 |

| Tier B, 19 LLM systems | P.1 | P.2 | P.3 |
|---|---|---|---|
| closed mean | 62.5 | 80.1 | 34.2 |
| open mean | 35.5 | 69.5 | 15.4 |
| **gap** | **+27.1** | **+10.5** | **+18.8** |
| sd across systems | 21.5 | 9.0 | 10.4 |

**Column meanings.**
- **P.1:** E is the executable prefix; S is state robustness.
- **P.2:** "parts on evidence" is the share of parts whose accepted set came from evidence rather than from the judgement.
- **P.3:** the columns are the mean $w_c$ per capability kind.
- **Fully operable:** $\min_c w_c$.

**Changes against the second pass (Tier B, closed mean):**
- P.1 went from 54.5 to **62.5**. The robustness term replaces the all-or-nothing prefix: the old version gave 0 to any order whose first step floats.
- P.2 went from 79.3 to **80.1**. Every part is now scored, not only the volume the evidence covered.
- P.3 went from 47.8 to **34.2**. Capabilities are now weakest-link chains instead of means of means. GPT-6 Astra drops from 53.2 to 39.3.

**What breaks the chains.** Of the F subsystems that score 0, 2,042 break on a missing named part (engine, brakes, springs, steering column, thermostat, …). 41 break on a power or motion path, including paths that end on a part that cannot move.
