# Where the recordings come from

None of the audio is redistributed in this repository. Fetch it from the
authors, cite them, and keep their licence.

## 1. Broiler chicks, healthy / unhealthy / noise (Mendeley Data `zp4nf2dxbh`)

Adebayo, S. (creator); contributors Aworinde, H., Akinwunmi, A., Alabi, O.,
Ayandiji, A., Oke, O., Oyebamiji, A., Adeyemo, A., Sakpere, A., Echetama, K.
(2023). *Poultry Vocalization Signal Dataset for Early Disease Detection.*
Mendeley Data, V1. doi:10.17632/zp4nf2dxbh.1. Bowen University. Licence
CC BY 4.0. (Author list taken from the DataCite record for the DOI; the
dataset page names the same people with their roles.)

346 mono WAV recordings in three folders: `Healthy` (139 files, 78 min),
`Unhealthy` (121 files, 58 min), `Noise` (86 files, 54 min). The file names
are integers and carry no day, pen or bird identity.

**Design, as the dataset page describes it** ("Steps to reproduce"): day-old
chicks at the Bowen University poultry research farm were divided into two
groups, one treated for respiratory disease and one not, "placed in a
controlled environment, separately from each other", each with its own
microphone; recording ran three times daily for 65 days; after 30 days the
untreated group developed respiratory disease, and those recordings were
labelled unhealthy. Two things follow for any classifier: the two classes
are two groups of birds in two rooms with two microphones, and the
unhealthy class is recorded only after day 30 while the healthy class spans
the whole 65 days — older birds against birds of every age.

**Metadata versus files:**

| the page says | the files are |
|---|---|
| 96 kHz, 24-bit | 48 kHz, 16-bit PCM |
| "collected and stored in MA4 and later converted to WAV" | WAV containers; the audio has been through a lossy codec at least once |
| 346 files | 346 files |

```
python -c "import urllib.request as u; u.urlretrieve('https://data.mendeley.com/public-api/zip/zp4nf2dxbh/download/1', 'data/zp4nf2dxbh-1.zip')"
```

Unzip into `data/mendeley_zp4nf2dxbh/{Healthy,Unhealthy,Noise}/*.wav` (the
top-level folder name in the archive contains a byte some `unzip` builds
reject; `prepare.py` does not need it). `manifest.sha256` holds the digest
of the archive as downloaded on 12 September 2026.

## 2. Layer-type pullets under acute stressors (Zenodo `10433023`)

Neethirajan, S. (2023). *Vocalization Patterns in Laying Hens — An Analysis
of Stress-Induced Audio Responses.* Zenodo. doi:10.5281/zenodo.10433023.
Licence CC BY 4.0. Described in the bioRxiv preprint of the same title
(doi:10.1101/2023.12.26.573338).

**Design, as the preprint describes it:** 52 Super Nick chickens in three
cages — two treatment cages of 20 birds (4 × 4 × 2 m) and one control cage
of 12 (4 × 2 × 2 m); observed from 3 days to 9 weeks of age; stressors
(a suddenly opened umbrella; recorded dog barking) applied on set days
between 14 and 64 days of age; one hour recorded before and one hour after
each stressor; the control cage recorded in protocol weeks 4 and 5; one
microphone two metres above the floor per cage; audio de-noised (iZotope RX)
and normalised before release.

**Files:** MP3, 44.1 kHz stereo, about 17.7 min each, named
`<Ctrl|Trt1|Trt2>_W<week>_D<day>_<PrS|PoS>[_<position>]`, where PrS/PoS
are before/after the stressor. Positions `_1.._4` exist for most cage-hours
and are four microphones recording the same birds at the same time — in
this repository they are one *session*, never split across train and test.

**Metadata versus files:**

| the record says | the files are |
|---|---|
| "laying hens" (title) | birds of 2–6 weeks by the preprint's own age range: pullets, not laying hens |
| stressors between 14 and 64 days of age | protocol weeks 1–5 in the file names; the mapping to age is not in the files. This repository assumes week 1 day 1 = day 14 (`AGE_OFFSET = 13`) and reports how the growth results move under other offsets |
| control cage "weeks 4 and 5" | `Ctrl_W4_*` and `Ctrl_W5_*`, consistent |
| 17.7 min per file, "one hour before and after" | each file is a 17.7-minute excerpt; which part of the hour is not stated |
| de-noised and normalised | the released spectra contain no energy below 2 kHz; whether that is the de-noising, a high-pass, or the codec is not stated |

```
https://zenodo.org/api/records/10433023/files/Laying%20Hens%20Vocalization%20Control%20Experiments%20Data.zip/content
https://zenodo.org/api/records/10433023/files/Laying%20Hens%20Vocalization%20Treatment%20Experiments%20Data.zip/content
```

Unzip both into `data/zenodo_10433023/`. Digests in `manifest.sha256`.

## 3. Single calls of backyard hens (GitHub `zebular13/ChickenLanguageDataset`)

A citizen-science collection of single vocalisations named by what the
recordist believed the bird meant, plus longer coop segments and noise.
The repository carries no licence file at the time of writing (September
2026). This repository therefore uses the recordings only to compute the
aggregate numbers in the f₀ survey (a median and a histogram per call
type), copies no audio, and redistributes nothing; if the maintainer
declines a licence, the survey line is removed.

```
git clone --depth 1 https://github.com/zebular13/ChickenLanguageDataset data/chicken_language
```
