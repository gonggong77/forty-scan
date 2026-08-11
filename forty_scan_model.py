# ==========================================================
# forty-scan
# 영포티 말투 판독기 - 모델 생성 스크립트
#
# 기반 문서:
# 영포티판독기_PRD_v2.0.md
#
# 모델:
# Logistic Regression (해석용 최종 모델)
# Decision Tree / Random Forest (성능 비교용)
#
# 특징추출(정규식 사전 + 오타 보정)은 forty_scan_features.py 를
# import 해서 씁니다. (테스트 스크립트와 동일한 전처리를 공유하기 위함)
#
# 기능:
# 1. CSV 불러오기 및 정리
# 2. Multi-hot 벡터 변환
# 3. Train / Test 분할 (stratify)
# 4. 3개 모델 학습 및 성능 비교
# 5. Accuracy / Precision / Recall / F1 평가
# 6. 특징별 Logistic Regression 계수 출력
# 7. 모델 저장
#
# ==========================================================


# ==========================================================
# 0. 라이브러리
# ==========================================================

import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from forty_scan_features import (
    ALIASES,
    FEATURES,
    FEATURE_NAMES,
    FUZZY_KEYWORDS,
    MODEL_DIR,
    MODEL_PATH,
    USE_TYPO_CORRECTION,
    transform_texts,
)


# 윈도우 콘솔에서 한글/이모지가 깨지지 않도록 출력 인코딩 지정
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==========================================================
# 1. 기본 설정
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "forty-scan_data.csv")

# 존댓말 보강 데이터 (일반 존댓말이 영포티로 오탐되는 것을 막기 위한 label=0 문장)
# 파일이 없으면 건너뛰므로 이 스크립트만 따로 받아도 동작합니다.
POLITE_PATH = os.path.join(BASE_DIR, "data", "forty-scan_data_polite.csv")

RANDOM_STATE = 42


# ==========================================================
# 2. 프로그램 시작
# ==========================================================

print()
print("=" * 70)
print("                        FORTY-SCAN")
print("                     영포티 말투 판독기 - 모델 생성")
print("             Multi-hot encoding + Logistic Regression")
print("=" * 70)


# ==========================================================
# 3. CSV 불러오기
# ==========================================================

print()
print("[1/6] 데이터 불러오는 중...")
print(f"      파일 경로 : {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "\nCSV 파일을 찾을 수 없습니다.\n"
        f"현재 경로:\n{DATA_PATH}\n\n"
        "DATA_PATH를 실제 파일 위치로 수정하세요."
    )

df = pd.read_csv(DATA_PATH, encoding="utf-8")

print(f"      기본 데이터 : {len(df)}행")

# ----- 존댓말 보강 데이터 병합 -----
#
# 원본 데이터는 label=0이 전부 대학생 반말 채팅이라
# "존댓말 = 영포티"로 학습되는 문제가 있습니다.
# 영포티가 아닌 존댓말(업무/공지/문의) 문장을 label=0으로 보강해
# 존댓말 자체가 영포티 신호가 되지 않도록 바로잡습니다.

if os.path.exists(POLITE_PATH):
    df_polite = pd.read_csv(POLITE_PATH, encoding="utf-8")
    df = pd.concat([df, df_polite], ignore_index=True)
    print(f"      존댓말 보강 : {len(df_polite)}행 추가")
else:
    print("      존댓말 보강 : 파일이 없어 건너뜁니다.")

before = len(df)

# 결측/공백 문장 제거 후 중복 문장 정리
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.drop_duplicates(subset="text").reset_index(drop=True)
df["label"] = df["label"].astype(int)

print(f"      전체 {before}행 → 정리 후 {len(df)}행 (중복/결측 {before - len(df)}행 제거)")
print(f"      일반(0)  : {int((df['label'] == 0).sum())}개")
print(f"      영포티(1): {int((df['label'] == 1).sum())}개")


# ==========================================================
# 4. Multi-hot 벡터 변환
# ==========================================================

print()
print("[2/6] Multi-hot 벡터로 변환하는 중...")

X = transform_texts(df["text"].tolist())
y = df["label"].to_numpy()

print(f"      벡터 행렬 크기 : {X.shape}  (문장 수 × 특징 수)")

coverage = (X.sum(axis=1) > 0).mean() * 100
print(f"      특징이 1개 이상 검출된 문장 비율 : {coverage:.1f}%")

# 예시로 첫 번째 영포티 문장의 벡터를 보여줍니다.
sample_idx = int(np.argmax(y == 1))
print()
print("      [변환 예시]")
print(f"      문장   : {df['text'].iloc[sample_idx]}")
print(f"      벡터   : {X[sample_idx].astype(int).tolist()}")


# ==========================================================
# 5. Train / Test 분할
# ==========================================================

print()
print("[3/6] 학습/검증 데이터 분할 중... (8:2, stratify=y)")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"      학습 데이터 : {X_train.shape[0]}개")
print(f"      검증 데이터 : {X_test.shape[0]}개")


# ==========================================================
# 6. 모델 학습 및 비교 (PRD 5.3 모델 후보 3종)
# ==========================================================

print()
print("[4/6] 모델 3종 학습 및 비교 중...")

candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

results = {}

for name, candidate in candidates.items():
    candidate.fit(X_train, y_train)
    pred = candidate.predict(X_test)
    results[name] = (accuracy_score(y_test, pred), f1_score(y_test, pred))

print()
print("      " + "-" * 46)
print(f"      {'모델':<22}{'Accuracy':>10}{'F1':>10}")
print("      " + "-" * 46)

for name, (acc, f1) in results.items():
    print(f"      {name:<22}{acc:>10.3f}{f1:>10.3f}")

print("      " + "-" * 46)

# 해석(계수 기반 기여도 출력)이 가능한 로지스틱 회귀를 최종 모델로 사용합니다.
model = candidates["LogisticRegression"]

print()
print("      최종 채택 모델 : LogisticRegression")
print("      (계수로 '어떤 특징이 점수를 올렸는지' 설명할 수 있기 때문)")


# ==========================================================
# 7. 성능 평가 (분류 4대 지표)
# ==========================================================

print()
print("[5/6] 최종 모델 성능 평가")
print()

y_pred = model.predict(X_test)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["일반(0)", "영포티(1)"],
        digits=3,
    )
)

# ----- 특징별 계수(가중치) 전체 출력 -----

coefficients = model.coef_[0]

print("      [특징별 가중치 - 값이 클수록 영포티 신호]")
print("      " + "-" * 46)

for name, coef in sorted(
    zip(FEATURE_NAMES, coefficients), key=lambda item: item[1], reverse=True
):
    direction = "영포티 ↑" if coef > 0 else "영포티 ↓"
    print(f"      {name:<18}{coef:>+8.2f}   {direction}")

print("      " + "-" * 46)


# ==========================================================
# 8. 모델 저장
# ==========================================================

print()
print("[6/6] 모델 저장 중...")

os.makedirs(MODEL_DIR, exist_ok=True)

# 오타 보정 설정도 함께 저장합니다.
# pkl을 불러 쓰는 쪽에서 학습 때와 똑같은 전처리를 재현해야 하기 때문입니다.
# (학습과 추론의 벡터화 방식이 다르면 계수가 엉뚱한 값에 곱해집니다)
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "use_typo_correction": USE_TYPO_CORRECTION,
        "fuzzy_keywords": FUZZY_KEYWORDS,
        "aliases": ALIASES,
    },
    MODEL_PATH,
)

print(f"      저장 완료 : {MODEL_PATH}")
