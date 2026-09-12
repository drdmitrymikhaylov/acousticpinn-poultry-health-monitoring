# What a poultry house says, and what an AI can hear in it

Reviews of AI in poultry report respiratory-disease and distress
classifiers at 94–99 % from audio. Almost all of those numbers are random
splits of clips from one house, and the same reviews name that as the
field's central weakness. This repository puts three open poultry
recordings — broiler chicks labelled healthy or unhealthy, layer pullets
recorded before and after an acute stressor, and single calls of backyard
hens — through the same question: **what survives a recording the model
has never heard?**

Three readings of the sound are compared under protocols that get harder.
A log-mel convolutional network (what the literature does); eight band
energies (what a chart shows); and a **physical model of the flock as a
population of source–filter voices** — a syrinx comb through a tracheal
tube over a background — which can only explain a spectrum through the
harmonics of a voice and returns numbers with names: the flock's
fundamental frequency, its spread, the share of the sound that is voiced,
the effective tract length, the background slope, and the rate of
broadband respiratory-type bursts. The same fundamental, tracked over
weeks, is then fitted with a growth law: **allometry and Gompertz growth
say how a flock's voice should fall with age**, and the page reports
whether it does.

---

## The data

None of it is redistributed here; `data/SOURCE.md` says where it lives
and who to cite.

| source | what | recordings | the group a clip belongs to |
|---|---|---|---|
| Mendeley `zp4nf2dxbh` (Adebayo et al. 2023) | broiler chicks, folders *Healthy* / *Unhealthy* / *Noise*, day-old to 65 d | 139 / 121 / 86 WAV, 48 kHz | the recording only — no day, pen or bird in the file names |
| Zenodo `10433023` (Neethirajan 2023) | white-egg layer pullets, an hour **before** and **after** an acute stressor (umbrella, dog barking) each week for five weeks; a control cage with no stressor | 70 + 32 MP3 of 17.7 min, released high-passed at 2 kHz | recording, and the protocol week |
| GitHub `zebular13/ChickenLanguageDataset` | single calls of backyard hens, named by meaning | 126 WAV | used only for the f₀ survey |

Two-second clips at 16 kHz, at most 60 per recording so that no
fifteen-minute file dominates: 3 439 broiler clips from 236
recordings, 4 200 pullet clips from 70 recordings in the stressed
cages, 1 920 from 32 in the control cage.

---

## What came out

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **The broiler dataset is separable by its recorder, not its birds.** The network scores AUC 1.00 on a random split and 1.00 on recordings it has never heard — and so does clip loudness alone (0.95). The two folders were recorded under different conditions; every reading of the sound, physical or not, tells them apart, and none of that is a disease detector. | §2 |
| 2 | **An acute stressor leaves almost nothing in an hour of flock sound.** Before-versus-after in the stressed cages: the network 0.69 on a random split, 0.63 on held-out recordings, **0.56 on a held-out week**. In the control cage, where nothing happened between "before" and "after", the same protocols give 0.52 / 0.56. The difference between the two — the stress signal — is at most a few hundredths of AUC. | §2 |
| 3 | No reading of the sound does better. Eight named physical numbers: 0.55 on held-out recordings, 0.54 on a held-out week; band energies 0.57 / 0.59; loudness 0.57 / 0.57; the control cage gives the same numbers. The physics is not rescuing a signal here; it is confirming, with numbers that have names, that there is little to rescue in these files. | §3 |
| 4 | **The flock's voice falls with age** — Spearman −0.57 over 102 recordings from 14 to 42 days — and a Gompertz-allometry curve with a *published* growth rate and inflection age and one free constant fits it with log-RMSE 0.178, the same as a straight line with two free numbers (0.177). Four weeks of a nine-week curve are not enough to tell a growth law from a slope; freed, the data ask for k = 0.035 /day and t_i = 18 d, which is not a pullet. | §4 |
| 5 | The model knows what a voice looks like; it does not know what a chicken is. In the broiler recordings the f₀ mass sits at 740 Hz — an adult's cluck, not a chick's peep — with class-spectrum peaks at exactly 250, 1000 and 2000 Hz; the released pullet files contain nothing below 2 kHz at all. A physical model makes both facts visible; a spectrogram network learns around them. | §3, §5 |

---

## 1. What the recordings sound like

![class spectra](figures/01_class_spectra.png)

Mean spectrum per class over recordings, with the 10th–90th percentile of
recordings shaded; the unit of replication is the recording. Two things
are visible before any model is fitted. In the broiler set the *Noise*
folder looks like the other two: the bird sound is a minority of the
power, and healthy and unhealthy differ across the whole band, including
where no chick vocalises. In the pullet set the released files are empty
below 2 kHz — the authors de-noised and high-passed before publishing —
so the only thing left is the chick's voice band, and after a stressor the
2–3 kHz share rises from 0.30 to 0.41 while the control cage moves from
0.28 to 0.32.

---

## 2. The same network, three ways of splitting the data

![protocols](figures/02_protocols.png)

A small convolutional network on log-mel spectrograms, trained the same
way under each protocol: **random** — five stratified folds over clips,
clips from the same recording on both sides (how most published results
are produced); **recording** — five folds over whole recordings; **week**
— leave one protocol week out (pullets). Folds under the last two can be
single-class, so out-of-fold scores are pooled and scored once.

| dataset · task | random | recording | week |
|---|---|---|---|
| broiler · healthy vs unhealthy, CNN | 1.00 | 1.00 | — |
| broiler · the same, clip loudness only | 0.95 | 0.95 | — |
| pullets, stressed cages · before vs after, CNN | 0.69 | 0.63 | 0.56 |
| pullets, control cage · "before" vs "after", CNN | 0.64 | 0.52 | 0.56 |

The broiler row is the finding the dataset's own paper does not report:
loudness alone separates the two folders on recordings it has never seen.
Whatever made the *Unhealthy* recordings different from the *Healthy* ones
— room, gain, day, microphone — is in every clip, and a classifier that
reaches 1.00 on it has learned that and nothing about respiratory disease.
The pullet rows are the honest version of the stress question: the
control cage, where the two labels are the same hour repeated, is the
false-positive rate of the whole method.

---

## 3. Putting the physics in: the flock as a population of voices

![source-filter model](figures/03_source_filter.png)

A chicken is a source–filter system. The syrinx sets a fundamental f₀ and
its harmonics; the trachea and open beak are a tube, closed at the syrinx
and open at the beak, with resonances at odd quarter-wavelengths,
F_m = (2m−1)·c/(4L). A clip of flock sound is written as a population of
such voices over a background, and nothing else:

    S(f) = g · [ ( Σₖ p(f₀ₖ) Σₕ a(h, f₀ₖ) L(f; h·f₀ₖ, ε·h·f₀ₖ) ) · |H_L(f)|²  +  B(f) ]

p(f₀) is the distribution of fundamentals in the clip (80 points on a log
grid from 250 Hz to 4 kHz — the day-old peep at the top, the adult cluck
at the bottom, and nothing a chicken does below 250 Hz: what is periodic
down there is the building, and it is left to the background); a(h, f₀)
is the harmonic profile of one voice, a small network shared by every
clip; the line width is a fixed *fraction* ε of the harmonic's frequency,
because pitch jitter is relative; |H_L|² is the three-formant tube filter
with one length L per clip and one quality factor shared by all; B(f) is
a power-law background per clip. The physics is in the model class — two
shared numbers (ε, Q) and one tract length per clip — not in a loss term.
The model is fitted to the spectra alone by log-spectral error, and
**the label never enters the fit**; under the recording and week
protocols the shared physics is refitted on the training clips and frozen
before the held-out clips are decomposed.

**What the fit finds.** A line width of 0.6 % of frequency, a tube Q of
6.8, and a harmonic profile that is flat until a harmonic leaves the
band — the network learned that harmonics above 7 kHz do not exist and
nothing else, which is the right amount of freedom for it to have. The
decomposition puts the backyard hens' single calls at a median f₀ of
734 Hz with a tract of 9.0 cm (a hen's trachea is of that order), the
pullets at 1290–1350 Hz, and the broiler recordings at 740 Hz — the last
is not a chick peep, and §5 says what it probably is. On the high-passed
pullet files the voiced share collapses to a few percent and the tract
length runs to its bounds: a model that needs the 250 Hz – 2 kHz band to
see a tube cannot see one in a file that has none.

**What the numbers are worth.** eight quantities per clip — f₀ mean and
spread, voiced share, tract length, background slope and level, and from
the waveform the count and energy share of broadband bursts of under 80 ms
in the 1–7.5 kHz band (the shape of a rale, a snick, or a beak on a pan;
`src/transients.py`) — go into a logistic regression under the same
protocols, next to the band energies, the full f₀ histogram, and the
network:

| reading of the sound (pullets, stressed cages) | random | recording | week |
|---|---|---|---|
| log-mel CNN | 0.69 | 0.63 | 0.56 |
| 8 band energies | 0.61 | 0.57 | 0.59 |
| f₀ histogram, 80 numbers | 0.56 | 0.55 | 0.56 |
| **physics, eight named numbers** | 0.57 | 0.55 | 0.54 |
| burst count and share only | 0.54 | 0.53 | 0.53 |
| control cage, physics (nothing happened) | 0.60 | 0.56 | 0.60 |

Read across a row and nothing separates from the control cage. The
network is the only reading that gains anything on the random split
(0.69), and it is also the only one that gains it on the *control* cage
(0.64), where the labels are the same hour repeated: what it learns is
"first hour of a recording day versus second", not stress. On a held-out
week every reading, physical or not, sits between 0.53 and 0.59 in both
cages. The physical numbers do not rescue a signal the other readings
missed; they show, in quantities with names, that one hour of flock sound
after an umbrella or a barking recording is not measurably different from
the hour before it in these files — and the recording-level view agrees:
the single most separating named number in the stressed cages is the
voiced share (AUC 0.66 over recordings), and it separates the control
cage's "before" and "after" just as well (0.67).

---

## 4. A growth curve read by microphone

![growth](figures/05_growth.png)

Across birds the fundamental of a call scales with body mass as
f₀ ∝ M^(−1/3) — longer trachea, heavier labia — and a growing flock's mass
follows a Gompertz law M(t) = M∞·exp(−exp(−k(t−t_i))). Together:

    f₀(t) = f₀∞ · exp( exp(−k (t − t_i)) / 3 )

with f₀∞ = a·M∞^(−1/3) the one number a microphone can identify on its
own; k and t_i are the growth parameters of the flock, and those a
hatchery publishes. The test: take k = 0.02 /day and t_i = 53 d from a
published Gompertz fit for white-egg layer pullets (Hy-Line; Oliveira et
al. 2018), leave one free constant, and see whether the flock's
fundamental over five weeks follows the curve.

| model of f₀ over age, 102 recordings | free numbers | log-RMSE |
|---|---|---|
| Gompertz + allometry, published k and t_i | 1 | 0.178 |
| Gompertz + allometry, all free (k = 0.035 /d, t_i = 18 d) | 3 | 0.176 |
| straight line in log f₀ | 2 | 0.177 |

The direction is right and the discrimination is absent. The fundamental
falls from about 1 300 Hz at two weeks to about 1 000 Hz at six, as
allometry says a growing bird's should, and the curve with the published
Hy-Line parameters passes through the data with one free number. But a
straight line in log f₀ does exactly as well, and the freed fit lands at
an inflection of 18 days and a rate of 0.035 /day — a curve for a bird
that finishes growing at five weeks, not a layer. Four protocol weeks
inside a nine-week growth curve do not contain the curvature that would
tell the two apart; the weeks that would (1–2 and 7–9) are the ones the
experiment did not record. What this section can claim is the sign and
the size of the trend; the growth law would need the whole cycle.

The right panel is the stress contrast at matched age: the median f₀ after
the stressor divided by the median before it, per cage and day, with the
control cage in grey. There is no consistent direction: nine of sixteen stressed-cage pairs go
up, seven go down, and the largest single jump (×1.7 at 35 days) is in
the control cage. Whatever the stressor did to the birds in the minutes
after it, an hour of recording has averaged it away.

---

## 5. Where the model puts the voices

![f0 survey](figures/04_f0_survey.png)

Left: the mean p(f₀) per source. The backyard single calls and the pullets
sit where a chicken should; the broiler recordings put most of their
voiced mass at 740 Hz, with the exact peaks of the class spectra
at 250, 1000 and 2000 Hz — round numbers no bird produces, and the
signature of a recorder or its power supply. The model has no way to
refuse a periodic source, and it says so: a distribution parked on a
harmonic series that is not a chick is visible in this panel, where a
spectrogram network would simply learn it as "healthy".

Right: the broiler file numbers against each recording's f₀. The dataset
says the birds were recorded from day-old over 65 days but carries no
dates; if the numbering followed time, f₀ would fall with it. It rises
with file number in the healthy folder (Spearman +0.49) and falls in the
unhealthy one (−0.67): the two folders were numbered in different orders,
or recorded in different sessions, and the file number carries no age
that the physics can use. This is the same lesson as §2 from the other
side — the structure in this dataset is the structure of its recording,
not of its birds.

---

## Verification

Thirteen checks in `tests/test_all.py`, all passing:

- a 4.5 s recording gives exactly two whole clips; a short single call is
  padded only when asked
- the mel filterbank covers 150 Hz – 7.5 kHz without a gap
- three 20 ms broadband bursts in two seconds are counted as 1.5 per
  second; a steady tone is not a transient
- a single voice at 1 kHz produces peaks only at integer multiples of
  1 kHz, and nowhere else
- an 8 cm closed–open tube puts its first formant at c/4L = 1072 Hz
- the line width scales with harmonic number
- a clip placed entirely at one fundamental is read back with zero spread
- **the results are pinned**: loudness alone must separate the broiler
  folders on held-out recordings (AUC > 0.9), the CNN must be near 1.0
  there, the pullet numbers must exist under leave-one-week-out, and the
  flock's fundamental must fall with age (Spearman below −0.5)

---

## What this does not show

- **No disease was detected here.** The broiler labels are separable by
  recording condition, and nothing in this repository can say whether the
  unhealthy birds also sounded different. A dataset with both classes
  recorded in the same room on the same days would answer that; this one
  cannot.
- **Three cages, one experiment.** Every pullet number rests on 70
  recordings from two stressed cages and 32 from one control cage, all
  in one building, released after de-noising and a 2 kHz high-pass that
  removed everything the source–filter model would have said about the
  tube. The AUCs are a report on this experiment, not on stress.
- **Age is a protocol week plus an assumption.** The preprint says the
  stressors ran from day 14; the file names give the protocol week and
  day, and the page takes day 1 of week 1 as 14 days old. The Gompertz
  comparison inherits that offset, and Super Nick is not Hy-Line.
- **The model cannot tell a bird from any other periodic source**, and
  the broiler set is the demonstration.
- **No house calibration.** Nothing here is in absolute units: no room
  constant, no fan noise floor, no sound power. That is what a deployment
  would add first, and what makes numbers comparable between houses.

---

## Source code

- `src/prepare.py` — recordings to labelled clips, with provenance
- `src/spectra.py` — class spectra and band shares at recording level
- `src/model.py`, `src/train.py` — the log-mel network and the three
  protocols, pooled scoring, resumable
- `src/transients.py` — the burst counter
- `src/syrinx_pinn.py` — the source–filter mixture, its named features,
  the f₀ survey and the protocol comparison
- `src/growth_pinn.py` — the Gompertz–allometry curve against published
  growth parameters
- `src/figures.py`, `tests/test_all.py`

`results/` holds every number on this page as JSON. `data/SOURCE.md`
says how to fetch the recordings; they are not redistributed.

```
pip install -r requirements.txt
python src/prepare.py && python src/spectra.py
python src/train.py                 # ~30 min on Apple silicon
python src/syrinx_pinn.py           # ~1 h on Apple silicon (MPS), longer on CPU
python src/growth_pinn.py && python src/figures.py && python tests/test_all.py
```

---

## Licence and credit

Documentation, figures and result files: CC BY 4.0. Source code in `src/`
and `tests/`: MIT.

The recordings belong to their authors. Cite them, not this page:

- Adebayo, S., Aworinde, H. O., Akinwunmi, A. O., et al. 2023. *Poultry
  Vocalization Signal Dataset for Early Disease Detection.* Mendeley Data
  V1, doi:10.17632/zp4nf2dxbh.1 (CC BY 4.0).
- Neethirajan, S. 2023. *Vocalization Patterns in Laying Hens — An
  Analysis of Stress-Induced Audio Responses.* Zenodo,
  doi:10.5281/zenodo.10433023 (CC BY 4.0); preprint
  doi:10.1101/2023.12.26.573338.
- zebular13, *ChickenLanguageDataset*, github.com/zebular13/ChickenLanguageDataset.
- Oliveira, C. F. S., et al. 2018. Mathematical models to describe the
  growth curves of white-egg layers. *Semina: Ciências Agrárias* 39(3):
  1327, doi:10.5433/1679-0359.2018v39n3p1327 — the growth parameters in §4.

---

*One of a series of physics-informed acoustic projects; see the profile
[README](https://github.com/drdmitrymikhaylov) for the others. The
beehive repository uses the same harmonic-comb decomposition and the same
three-protocol test.*
