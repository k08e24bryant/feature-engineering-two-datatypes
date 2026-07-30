# EDA & Feature Engineering on Two Data Types

Exploratory Data Analysis and Feature Engineering applied to **two fundamentally different data types** — tabular and text — with a before/after evaluation on machine learning models to measure whether the engineering actually helped.

> Group assignment for the Big Data course, Informatics Engineering, Institut Teknologi Sepuluh Nopember (ITS).

## TL;DR

| Track | Dataset | Task | Model | Baseline | Engineered | Verdict |
|---|---|---|---|---|---|---|
| Tabular | Car Features & MSRP (11,914 × 16) | Price regression | Random Forest | MAE $3,884 · R² 0.940 | **MAE $3,067 · R² 0.974** | Aggressive FE wins (−21% error) |
| Text | SMS Spam Collection (5,572 msgs) | Spam classification | Naive Bayes / LogReg | F1 0.9514 | **F1 0.9559** (cleaning only) | Minimal targeted FE wins; over-engineering hurt |

**Core principle:** every feature transformation must trace back to a specific EDA finding, and every improvement claim must be proven against a raw-feature baseline.

## Repository Structure

```
├── 01_tabular.ipynb    # Car MSRP: EDA → FE → Random Forest before/after
├── 02_text.ipynb       # SMS Spam: EDA → FE → 6-configuration ablation study
├── slides/             # Presentation (PPTX)
└── README.md
```

## Track 1 — Tabular: Car Price Prediction

**EDA findings → FE actions:**

| EDA finding | Action |
|---|---|
| Target skewness 11.77 (supercars up to $2M) | Train on `log1p(MSRP)` |
| Price cluster at ~$2,000 = pre-2000 cars | `Car_Age` derived feature |
| `highway MPG` ↔ `city mpg` correlate 0.89; one impossible value (354) | Merge into `Avg_MPG`; error → NaN |
| `Engine HP` ↔ `Cylinders` correlate 0.78 | `HP_per_Cylinder` ratio (EVs with 0 cylinders handled) |
| `Market Category` 31% missing | Missingness as signal: `Is_Luxury`, `Is_Performance` flags |
| `Make` has 48 brands | Frequency encoding |
| `Popularity` r ≈ −0.05 with price | Dropped |

**Result:** 7 raw features → 46 engineered. MAE −21%, R² 0.940 → 0.974. Four of the top-five most important features are engineered ones (`Car_Age`, `Make_Freq`, `Is_Luxury`, `HP_per_Cylinder`).

## Track 2 — Text: SMS Spam Detection

**EDA findings:** classes imbalanced 87:13 (→ F1 as primary metric), spam is 3× longer, spam vocabulary is distinctive (*free, call, claim, prize*), digits/capitals are strong signals, and broken HTML entities (`&gt;` `&lt;`) leak into top tokens.

**Ablation study (isolating one change at a time):**

| # | Pipeline | Accuracy | F1 (spam) |
|---|---|---|---|
| 1 | Raw + BoW + NB (baseline) | 0.9874 | 0.9514 |
| 2 | **Clean + BoW + NB** | **0.9883** | **0.9559** |
| 3 | Clean + BoW bigram + NB | 0.9874 | 0.9527 |
| 4 | Clean + TF-IDF bigram + NB | 0.9740 | 0.8922 |
| 5 | Clean + TF-IDF bigram + manual + NB | 0.9812 | 0.9242 |
| 6 | Clean + TF-IDF bigram + manual + LogReg | 0.9857 | 0.9463 |

**Key insight:** TF-IDF *hurts* Naive Bayes here — it down-weights frequent words, but the frequency of *free*/*call* **is** the spam signal, and MultinomialNB is designed for counts. EDA-driven cleaning alone (HTML unescape + normalizing numbers/phones into tokens instead of deleting them) beat everything.

## Lessons Learned

1. **EDA is not decoration** — every effective feature traced back to a specific finding.
2. **Always keep a baseline** — it is the only way to prove (or disprove) that FE helped.
3. **Validate the pipeline itself** — a regex ordering bug silently deleted our normalized tokens (uppercase tokens removed by a lowercase-only filter); the code ran without errors and only a sanity-check print exposed it.
4. **FE is not one-size-fits-all** — aggressive FE won on tabular data; on short text, targeted minimal FE won and stacking techniques backfired.

## Running the Notebooks

Both notebooks run top-to-bottom on Google Colab with zero setup — datasets are loaded directly from public URLs.

```
pandas · numpy · scikit-learn · matplotlib · seaborn · wordcloud · scipy
```

Local run: `pip install -r` the packages above (all pre-installed on Colab), then execute cells in order.

## Data Sources

- [Car Features & MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset) — Kaggle (CooperUnion)
- [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — UCI Machine Learning Repository
