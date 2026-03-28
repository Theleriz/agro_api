import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(title="ML Scoring Service", version="5.0")

# Artificat(aka model) loading
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "scoring_artifact_v5.joblib")

try:
    artifact = joblib.load(ARTIFACT_PATH)
    print(f"Artifact alreadey loaded, artifcat version: {artifact.get('version', 'unknown')}")
except FileNotFoundError:
    print(f"File {ARTIFACT_PATH} not found")
    artifact = None


# Pydentic part
class ApplicationIn(BaseModel):
    app_num: str
    oblast: str
    district: str
    sector: str
    subsidy_name: str
    amount: float
    norm: float
    date: str
    status: str


class ApplicationOut(BaseModel):
    app_num: str
    oblast: str
    raw_score: float
    score: float
    tier: str


# features
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])

    d['month']          = d['date'].dt.month
    d['quarter']        = d['date'].dt.quarter
    d['day_of_week']    = d['date'].dt.dayofweek
    d['hour']           = d['date'].dt.hour
    d['days_since_jan'] = d['date'].dt.dayofyear

    d['is_early_year']  = d['month'].isin([1, 2]).astype(int)
    d['is_budget_end']  = d['month'].isin([11, 12]).astype(int)
    d['is_monday']      = (d['day_of_week'] == 0).astype(int)
    d['is_biz_hours']   = d['hour'].between(9, 17).astype(int)

    d['head_count']      = (d['amount'] / d['norm']).clip(0, 100_000)
    d['head_count_log']  = np.log1p(d['head_count'])
    d['amount_log']      = np.log1p(d['amount'])
    d['norm_log']        = np.log1p(d['norm'])
    d['amount_per_head'] = d['amount'] / (d['head_count'] + 1)

    d['is_import']   = (d['norm'] >= 390_000).astype(int)
    d['is_premium']  = (d['norm'] >= 525_000).astype(int)

    def norm_tier(n):
        if n <= 100:        return 'per_unit_small'
        elif n <= 5_000:    return 'per_unit_mid'
        elif n <= 50_000:   return 'small_subsidy'
        elif n <= 200_000:  return 'mid_subsidy'
        elif n <= 400_000:  return 'large_domestic'
        else:               return 'premium_import'

    d['norm_tier'] = d['norm'].apply(norm_tier)

    def subsidy_group(name):
        n = str(name).lower()
        if 'племен' in n and 'приобрет' in n: return 'purchase_breeding'
        elif 'молодняк' in n: return 'young_stock'
        elif 'молок' in n:    return 'milk'
        elif 'мяс' in n:      return 'meat'
        elif 'корм' in n:     return 'feed'
        elif 'осемен' in n:   return 'insemination'
        elif 'шерст' in n:    return 'wool'
        elif 'мёд' in n:      return 'honey'
        else:                 return 'other'

    d['subsidy_group'] = d['subsidy_name'].apply(subsidy_group)

    livestock = [
        'Субсидирование в скотоводстве',
        'Субсидирование в овцеводстве',
        'Субсидирование в верблюдоводстве',
        'Субсидирование в коневодстве',
    ]
    d['is_livestock'] = d['sector'].isin(livestock).astype(int)

    d['sector_x_oblast'] = d['sector'].astype(str) + '||' + d['oblast'].astype(str)

    region_avg = d.groupby('oblast')['amount'].transform('mean')
    d['amount_vs_region'] = d['amount'] / (region_avg + 1)

    return d


def score_applications(records: List[dict]) -> List[dict]:
    if artifact is None:
        raise RuntimeError("Артефакт модели не загружен")

    df = pd.DataFrame(records)

    # Шаг 2 — базовые признаки
    df = build_base_features(df)

    # Шаг 3 — target encoding
    gm = artifact['global_mean']
    df['region_exec_rate']   = df['oblast'].map(artifact['region_map']).fillna(gm)
    df['sector_exec_rate']   = df['sector'].map(artifact['sector_map']).fillna(gm)
    df['district_exec_rate'] = df['district'].map(artifact['district_map']).fillna(gm)
    df['sxa_exec_rate']      = df['sector_x_oblast'].map(artifact['sxa_map']).fillna(gm)

    # Шаг 4 — raw_score от модели
    X = artifact['preprocessor'].transform(df[artifact['all_features']])
    df['raw_score'] = (artifact['model'].predict_proba(X)[:, 1] * 100).round(2)

    # Шаг 5 — нормализация внутри региона
    df['score'] = (
        df.groupby('oblast')['raw_score']
        .transform(lambda x: (x.rank(method='min', pct=True) * 100).round(1))
    )

    # Шаг 6 — тиры
    thr = artifact['thresholds']
    df['tier'] = df['score'].apply(
        lambda s: 'Приоритет' if s >= thr['priority'] else
                  'Высокий'   if s >= thr['high']     else
                  'Средний'   if s >= thr['medium']   else 'Низкий'
    )

    result = (
        df[['app_num', 'oblast', 'raw_score', 'score', 'tier']]
        .sort_values('score', ascending=False)
        .to_dict(orient='records')
    )
    return result


# endpoints
@app.get("/health")
def health():
    version = artifact.get('version', 'unknown') if artifact else 'not loaded'
    return {"status": "ok", "model_version": version}


@app.post("/score", response_model=List[ApplicationOut])
def score(applications: List[ApplicationIn]):
    if artifact is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Проверь наличие scoring_artifact_v5.joblib")

    if not applications:
        raise HTTPException(status_code=400, detail="Список заявок пустой")

    records = [app.model_dump() for app in applications]

    try:
        result = score_applications(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка скоринга: {str(e)}")

    return result