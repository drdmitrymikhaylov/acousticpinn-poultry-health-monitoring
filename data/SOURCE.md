# Where the recordings come from

None of the audio is redistributed in this repository. Fetch it from the
authors, cite them, and keep their licence.

## 1. Broiler chicks, healthy / unhealthy / noise (Mendeley Data `zp4nf2dxbh`)

Adebayo, S., Aworinde, H. O., Akinwunmi, A. O., Alabi, O. M., Ayandiji, A.,
Sakpere, A. B., Oyebamiji, A. K., Olaide, O., Kizito, E., Olawuyi, A. J.
(2023). *Poultry Vocalization Signal Dataset for Early Disease Detection.*
Mendeley Data, V1. doi:10.17632/zp4nf2dxbh.1. Licence CC BY 4.0.

346 mono WAV recordings at 48 kHz, 16 bit (the dataset page says 96 kHz /
24 bit; the files say otherwise), in three folders: `Healthy` (139 files,
78 min), `Unhealthy` (121 files, 58 min), `Noise` (86 files, 54 min). The
description says the birds were recorded from day-old over 65 days; the
file names are integers and carry no day, pen or bird identity.

```
python -c "import urllib.request as u; u.urlretrieve('https://data.mendeley.com/public-api/zip/zp4nf2dxbh/download/1', 'data/zp4nf2dxbh-1.zip')"
```

Unzip into `data/mendeley_zp4nf2dxbh/{Healthy,Unhealthy,Noise}/*.wav` (the
top-level folder name in the archive contains a byte some `unzip` builds
reject; `prepare.py` does not need it).

## 2. Layer-type pullets under acute stressors (Zenodo `10433023`)

Neethirajan, S. (2023). *Vocalization Patterns in Laying Hens — An Analysis
of Stress-Induced Audio Responses.* Zenodo. doi:10.5281/zenodo.10433023.
Licence CC BY 4.0. Described in the bioRxiv preprint of the same title
(doi:10.1101/2023.12.26.573338): 52 Super Nick chickens in three cages
(two treatment cages of 20, one control cage of 12), recorded from 3 days
to 9 weeks of age; one hour before and one hour after a stressor each week
(a suddenly opened umbrella; recorded dog barking), control cage recorded
in weeks 4 and 5. One microphone two metres above the floor per cage. The
released files are MP3, 44.1 kHz stereo, about 17.7 min each, and were
de-noised and normalised by the authors before release.

```
https://zenodo.org/api/records/10433023/files/Laying%20Hens%20Vocalization%20Control%20Experiments%20Data.zip/content
https://zenodo.org/api/records/10433023/files/Laying%20Hens%20Vocalization%20Treatment%20Experiments%20Data.zip/content
```

Unzip both into `data/zenodo_10433023/`. File names such as
`Ctrl_W4_D1_PoS_1.mp3` carry condition, week, day, phase (PrS = before the
stressor, PoS = after) and position.

## 3. Single calls of backyard hens (GitHub `zebular13/ChickenLanguageDataset`)

A citizen-science collection of single vocalisations named by what the
recordist believed the bird meant, plus longer coop segments and noise.
No formal licence file at the time of writing; used here only for the
fundamental-frequency survey, and no audio is copied.

```
git clone --depth 1 https://github.com/zebular13/ChickenLanguageDataset data/chicken_language
```
