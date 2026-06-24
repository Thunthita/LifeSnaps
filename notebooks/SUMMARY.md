# LifeSnaps Calories Regression — Summary

`01_calories_regression.ipynb`

---

## Dataset
- **Source**: LifeSnaps (Fitbit + SEMA app) — 71 users, April 2021 – January 2022
- **Final size**: 6,638 rows (dropped rows where Calories < 500 kcal)
- **Drop justification**: Values < 500 kcal/day are below any physiologically possible human energy expenditure. The Harris-Benedict equation gives a minimum BMR of ~941 kcal/day (female, 40 kg, 150 cm, age 80). Ref: Harris & Benedict (1918), *A Biometric Study of Human Basal Metabolism*.
- **Days per user**: min=58, median=81, max=243
- **Target**: `calories` (daily total calorie burn from Fitbit, unit = **kcal**)

---

## 1. Pipeline

```
Raw Data (LifeSnaps CSV)
  ↓
Cohort Filter — drop rows where calories < 500 kcal
  ↓
Feature Engineering (39 features)
  ↓
Imputation — PerUserMedianImputer (per-user median, train only)
  ↓
Scaling — StandardScaler
  ↓
Models (7 algorithms)
  ↓
Hyperparameter Tuning — Optuna
  ↗ uses GroupKFold internally to score each trial
  ↓
Feature Selection — importance ≥ 1%
  ↓
Evaluation (multiple strategies)
```

---

## 2. Feature Engineering

39 features across 8 groups:

| Group | Features |
|---|---|
| Activity (raw) | steps, distance, very/moderately/lightly_active_minutes, sedentary_minutes |
| Activity (engineered) | TotalActiveMinutes, ActiveRatio, StepsPerActiveMin, DayOfWeek, IsWeekend, mindfulness_session |
| Heart rate | bpm, resting_hr, hr_zone_below/moderate/cardio/peak |
| HRV / Advanced | nremhr, rmssd, full_sleep_breathing_rate, filteredDemographicVO2Max |
| Skin temperature | nightly_temperature, daily_temperature_variation |
| Sleep | sleep_duration, minutesToFallAsleep, minutesAsleep, minutesAwake, minutesAfterWakeup, sleep_efficiency, sleep_deep/rem/light/wake_ratio |
| Demographics | age, bmi, gender_binary |
| Mood (SEMA) | ALERT, HAPPY, NEUTRAL, RESTED/RELAXED, SAD, TENSE/ANXIOUS, TIRED |
| Location (SEMA) | ENTERTAINMENT, GYM, HOME, HOME_OFFICE, OTHER, OUTDOORS, TRANSIT, WORK/SCHOOL |

---

## 3. Scaling — StandardScaler vs RobustScaler

### Feature Distribution (continuous features only)

Features ที่มี outlier/skew สูง (ควร RobustScaler):

| Feature | Skew | Outlier % |
|---|---|---|
| minutesToFallAsleep | 32.4 | 0.6% |
| hr_zone_peak | 18.6 | 9.8% |
| minutesAfterWakeup | 8.1 | 22.2% |
| hr_zone_cardio | 5.1 | 15.6% |
| sleep_efficiency | -5.0 | 4.2% |
| very_active_minutes | 3.6 | 8.8% |

### ผลการทดสอบ StandardScaler vs RobustScaler (Ridge & Lasso, GroupKFold)

| Scaler | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| StandardScaler | Ridge | 324.47 | 456.32 | 0.5506 |
| RobustScaler | Ridge | 324.00 | 455.70 | 0.5522 |
| StandardScaler | Lasso | 323.22 | 450.68 | 0.5626 |
| RobustScaler | Lasso | 321.35 | 448.48 | 0.5668 |

**สรุป:** ผลต่างเล็กน้อยมาก (~1-2 kcal) → คง **StandardScaler** ไว้

### Scaling กับแต่ละ Model

| Model | Relies on Scaling | เหตุผล |
|---|---|---|
| Linear Regression | Yes | OLS — scale ส่งผลต่อ coefficients |
| Ridge | Yes | L2 penalty — scale ส่งผลต่อ regularization |
| Lasso | Yes | L1 penalty — features ที่ไม่ scale ถูก penalize ไม่ยุติธรรม |
| Random Forest | **No** | Tree splits — scale invariant |
| Gradient Boosting | **No** | Tree-based — scale invariant |
| XGBoost | **No** | Tree-based — scale invariant |
| LightGBM | **No** | Tree-based — scale invariant |

> Scaling จึงมีผลจริงๆ เฉพาะ **Ridge และ Lasso** เท่านั้น

---

## 4. Cross-Validation Strategies

### ทุก Strategy ที่ใช้

| # | Strategy | ประเภท | จุดประสงค์ |
|---|---|---|---|
| 1 | **KFold** (3-fold) | CV — random | Baseline, leaky (users ปนกัน) |
| 2 | **GroupKFold** (3-fold, by user) | CV — user-aware | Predict user ใหม่ ใช้ใน Optuna tuning |
| 3 | **TimeSeriesSplit** | CV — temporal | Predict อนาคต, no temporal leakage |
| 4 | **Within-User (single split)** | Fixed split | Shared model, user เดิม last 20% = test |
| 5 | **Per-User (single split)** | Fixed split | Model แยกต่อ user, last 20% = test |
| 6 | **Within-User + TSCV** | CV per fold | แก้ leakage ของ WU-Single, retrain ต่อ fold |
| 7 | **Per-User + TSCV** | CV per fold | Per-user model ที่ reliable กว่า single split |
| 8 | **LOSO** (n=71 folds) | CV — มาตรฐาน paper | แต่ละ user เป็น test fold — standard ใน wearable research |

### Leakage Gap (KFold vs GroupKFold)

| Model | KFold MAE | GKF MAE | Leakage Gap |
|---|---|---|---|
| LightGBM | 105.7 | 290.8 | +185.1 |
| XGBoost | 108.6 | 291.3 | +182.7 |
| Random Forest | 109.4 | 288.5 | +179.1 |
| Gradient Boosting | 155.4 | 281.5 | +126.2 |
| Lasso | 253.0 | 323.2 | +70.2 |
| Ridge | 252.2 | 324.5 | +72.3 |
| Linear Regression | 252.1 | 323.7 | +71.6 |

> Tree models มี gap ใหญ่กว่า linear models มาก — overfit user identity ใน KFold

### GroupKFold vs LOSO

| | GroupKFold (k=3) | LOSO (k=71) |
|---|---|---|
| n_folds | 3 | 71 |
| test set per fold | users หลายคนรวมกัน | user เดียว |
| พบใน paper | น้อย | **มากที่สุดใน wearable research** |
| variance estimate | สูง (fold น้อย) | ต่ำกว่า (fold มาก) |

### Within-User & Per-User — Single Split vs TSCV

| Model | WU-Single | WU-TSCV | PU-Single | PU-TSCV |
|---|---|---|---|---|
| Linear Regression | 257.5 | 307.8 | 100.4 | 137.6 |
| Ridge | 257.6 | 306.1 | 81.4 | 88.2 |
| Lasso | 259.0 | 303.3 | 70.3 | **69.6** |
| Random Forest | 115.4 | 183.3 | 110.3 | 119.2 |
| Gradient Boosting | 164.9 | 201.3 | 99.8 | 106.0 |
| XGBoost | 120.4 | 180.9 | 106.7 | 117.6 |
| LightGBM | 125.2 | 177.5 | 161.0 | 260.2 |

**ข้อสังเกต:**
- Per-User: **Lasso ดีที่สุด** (MAE = 69.6) — linear model ชนะ tree-based เพราะ data ต่อ user น้อย
- LightGBM แย่มากใน Per-User (MAE = 260.2) — overfit รุนแรง
- WU-TSCV สูงกว่า WU-Single เพราะ retrain ต่อ fold → honest กว่า

---

## 5. Feature Selection

52 → **19 features** ที่ importance ≥ 1%:

| Group | Features |
|---|---|
| Activity | very_active_minutes, moderately_active_minutes, TotalActiveMinutes, distance, steps, StepsPerActiveMin, ActiveRatio, lightly_active_minutes |
| Heart rate | bpm, resting_hr, hr_zone_cardio |
| HRV | rmssd, nremhr |
| Advanced Fitbit | filteredDemographicVO2Max |
| Temperature | nightly_temperature, daily_temperature_variation |
| Demographics | bmi |
| Mood | ALERT, TIRED |

XGBoost GKF-tuned: 52 features (MAE=263.0) vs 19 features (MAE=265.4) → ต่างกันแค่ **+2.4 kcal**

---

## 6. Future Work

### 1) Outlier Removal — Box Plot แทนเกณฑ์ 500 kcal

ปัจจุบัน filter ด้วย `calories < 500 kcal` ซึ่งเป็นเกณฑ์ตายตัว (hard threshold)

**ที่ควรลองทำต่อ:**
- Plot box plot ของ `calories` แยกตาม user หรือ gender
- ตัด outlier ด้วย IQR method แทน เช่น `Q1 - 1.5×IQR` และ `Q3 + 1.5×IQR`
- เปรียบเทียบ MAE ก่อน/หลังตัด outlier แบบ data-driven

```python
Q1 = df['calories'].quantile(0.25)
Q3 = df['calories'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['calories'] >= Q1 - 1.5*IQR) & (df['calories'] <= Q3 + 1.5*IQR)]
```

**ข้อดี:** ไม่ต้องสมมติเกณฑ์เอง ใช้ distribution ของข้อมูลจริงแทน

### 2) Age — ลอง One-Hot Encoding

ปัจจุบัน `age` ถูกตัดออกจาก 15 features เพราะ importance ต่ำเมื่อ encode เป็น binary (0/1)

**ที่ควรลองทำต่อ:**
- One-Hot encode `age` (`"<30"` → `age_lt30`, `">=30"` → `age_gte30`) แทน binary
- เพิ่มเข้า feature set แล้วเปรียบเทียบ MAE กับ 15 features เดิม

```python
df = pd.get_dummies(df, columns=['age'], prefix='age')
# ได้คอลัมน์ใหม่: age_<30, age_>=30
```

**หมายเหตุ:** `age` มีแค่ 2 ค่า → One-Hot กับ binary encoding ให้ผลเหมือนกัน แต่ถ้า dataset อนาคตมีช่วงอายุละเอียดกว่า (`<20`, `20-30`, `30-40`, `>=40`) One-Hot จะมีประโยชน์ชัดเจนกว่า

### 3) Scaling vs No Scaling — เปรียบเทียบ MAE

ปัจจุบันใช้ StandardScaler กับทุก model แต่ tree-based models (RF, XGBoost, LightGBM, GradientBoosting) ไม่จำเป็นต้องใช้ scaling เพราะ split-based ไม่ขึ้นกับ scale

**ที่ควรลองทำต่อ:**
- ทดสอบ 3 แบบ: `StandardScaler` vs `RobustScaler` vs `No Scaling`
- เปรียบเทียบ MAE แยกตาม model group (linear vs tree-based)

| Model Group | คาดว่า Scaling ช่วยไหม? |
|---|---|
| Linear, Ridge, Lasso, ElasticNet | ✅ ช่วยมาก |
| RF, GBM, XGBoost, LightGBM | ❌ แทบไม่ต่าง |

```python
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

# With scaling
pipe_scaled = Pipeline([('imputer', imputer), ('scaler', StandardScaler()), ('model', model)])

# No scaling
pipe_noscale = Pipeline([('imputer', imputer), ('model', model)])
```

**ประโยชน์:** ลด complexity ของ pipeline สำหรับ tree-based models และยืนยันว่า StandardScaler ที่ใช้อยู่จำเป็นจริงๆ หรือเปล่า

### 4) Include Gender ใน 15 Features แล้ว Retrain

จากการทดสอบเบื้องต้น พบว่าการเพิ่ม `gender_binary` เข้าไปใน feature set ช่วยลด MAE ได้อย่างมีนัยสำคัญ

| Feature Set | GKF MAE |
|---|---|
| 15 features (ไม่มี gender) | 287 kcal |
| 16 features (+ gender) | **219 kcal** |
| ดีขึ้น | **-68 kcal (-24%)** |

**ที่ควรทำต่อ:**
- เพิ่ม `gender_binary` (`MALE` → 1, `FEMALE` → 0) เข้าใน `feature_cols` ใน `train.py`
- Retrain และ save models ใหม่ทั้งหมด
- เปรียบเทียบ MAE ครบทุก 7 strategies ไม่ใช่แค่ GKF

```python
df['gender_binary'] = (df['gender'] == 'MALE').astype(int)

feature_cols = [
    'steps', 'distance', 'very_active_minutes', 'moderately_active_minutes',
    'lightly_active_minutes', 'TotalActiveMinutes', 'ActiveRatio', 'StepsPerActiveMin',
    'bpm', 'resting_hr', 'hr_zone_cardio', 'filteredDemographicVO2Max',
    'nightly_temperature', 'daily_temperature_variation', 'bmi',
    'gender_binary',  # เพิ่มใหม่
]
```

**เหตุผลที่ gender ช่วย:** GKF test บน user ใหม่ที่ไม่เคยเห็น — gender เป็น strong prior สำหรับ baseline calories (Male ~2,418 kcal vs Female ~1,850 kcal ต่างกัน ~568 kcal)

---

## 7. References

### LifeSnaps Dataset
- Yfantidou et al. (2022) — *"LifeSnaps, a 4-month multi-modal dataset capturing unobtrusive snapshots of our lives in the wild"* — **Scientific Data (Nature)**

### LOSO — มาตรฐานใน wearable papers
- Ellis et al. (2014) — RF + LOSO บน accelerometer — *Physiological Measurement*
- O'Driscoll et al. (2021) — RF, GBM บน Fitbit/ActiGraph, LOSO เป็น primary validation — *JMIR mHealth and uHealth*
- Staudenmayer et al. (2009) — ANN + LOSO บน ActiGraph 60 subjects — *Journal of Applied Physiology*
- Lee & Lee (2024) — CNN-LSTM + LOSO — *Sensors (MDPI)*

### Random Split — leaky baseline
- Dehghani et al. (2019) — เปรียบเทียบ random k-fold vs LOSO พบ random inflate accuracy ~10-16% — *arXiv:1904.02666*
- O'Driscoll et al. (2021) — run ทั้ง random split และ LOSO แล้ว show ว่า random optimistic — *JMIR*

### Within-Subject CV
- Scheurer et al. (2020) — person-specific (within-subject) vs LOSO พบ person-specific ดีกว่า 43.5% — *Sensors (MDPI)*
- Montoye et al. (2018) — LOOCV within-sample vs out-of-sample — *Journal of Applied Physiology*

### Survey เปรียบเทียบทุก strategy
- Rehman et al. (2024) — benchmark k-fold vs LOSO vs within-subject บน dataset เดียวกัน — *Algorithms (MDPI)*
- Álvarez-García et al. (2020) — survey EE estimation จาก wearables + discuss validation methodology — *ACM Computing Surveys*
