# ==========================================================
# forty-scan
# 영포티 말투 판독기
#
# 기반 문서:
# 영포티판독기_PRD_v2.0.md
#
# 모델:
# Logistic Regression (해석용 최종 모델)
# Decision Tree / Random Forest (성능 비교용)
#
# 특징 인코딩:
# Multi-hot encoding (등장 횟수를 그대로 값으로 사용)
#
# 0회 → 0
# 1회 → 1
# 3회 → 3
#
# 기능:
# 1. CSV 불러오기 및 정리
# 2. 28개 특징 정규식 사전
# 3. Multi-hot 벡터 변환
# 4. Train / Test 분할 (stratify)
# 5. 3개 모델 학습 및 성능 비교
# 6. Accuracy / Precision / Recall / F1 평가
# 7. 특징별 Logistic Regression 계수 출력
# 8. 모델 저장
# 9. 새로운 문장 반복 판독
# 10. 검출된 특징 / 등장 횟수 / 기여도 출력
# 11. 5단계 등급 판정
#
# ==========================================================


# ==========================================================
# 0. 라이브러리
# ==========================================================

import os
import re
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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

MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "forty_scan_현경_model.pkl")

RANDOM_STATE = 42


# ==========================================================
# 2. 특징 사전 (28개 차원)
# ==========================================================
#
# PRD 5.2 / 5.3 기준으로 확정한 28개 특징입니다.
#
# data/forty-scan_feature.csv 의 25개 리터럴 키워드는
# 아래 카테고리 정규식 안에 모두 흡수했습니다.
#
# 예) 미인 / 미녀 / 늘씬 / 아름다우셔서  →  외모_칭찬
#     오빠로써 / 인생선배로써 / 살아보니까 →  인생선배_훈수
#
# 앞의 23개는 영포티 신호,
# 뒤의 5개는 젊은 세대(일반) 신호입니다.
# 일반 신호는 로지스틱 회귀에서 음(-)의 계수를 받아
# 영포티 지수를 낮추는 방향으로 작동합니다.
#
# ==========================================================

FEATURES = [
    # ----- 영포티 신호 : 표기 습관 -----
    ("웃음_ㅎㅎ", r"ㅎ{2,}"),
    ("이모티콘_^^", r"\^\^|\^-\^"),
    ("물결_남용", r"~"),
    ("느낌표", r"!"),
    ("말줄임표", r"\.{2,}|…"),
    # 존댓말 자체는 세대 신호가 아니므로, 존댓말 뒤에 물결/이모티콘이 붙는
    # '영포티식 존댓말'만 잡습니다. (예: 하셨네요~ / 어떠신가요~^^)
    ("존댓말_물결조합", r"(?:요|죠|네요|세요|어요|아요)\s*[~^]|~\s*\^\^"),
    ("쉼표_남용", r",{2,}"),
    ("이모지_특수기호", r"[\U0001F300-\U0001FAFF☀-➿♥♡❤]"),

    # ----- 영포티 신호 : 어휘 -----
    ("오빠_호칭", r"오빠|옵빠|누나"),
    ("외모_칭찬", r"처자|미인|미녀|늘씬|아름다우|아름다운|예쁘|이쁘|동안|훈남"),
    ("인연_운명", r"인연|운명|만남 시작"),
    ("인생선배_훈수", r"인생선배|오빠로써|로써|살아보니까|살다 보니|조언"),
    ("라떼_나때", r"라떼|나 때|우리 때|내가 젊|왕년"),
    ("요즘애들", r"요즘 애들|요즘 것들|요즘 젊은|요즘 친구"),
    ("젊음_강조", r"젊음|젊은 애들|젊은 남자|핫바리|젊게"),
    ("옛날_유행어", r"핫플레이스|고고씽|방가|열정|대쉬|전번|므흣|짱"),
    ("감탄사_에혀", r"에혀|어허|허허|아이고|어이쿠|허참"),

    # ----- 영포티 신호 : 문장 습관 -----
    ("격식체_습니다", r"습니다|입니다|합니다|니다"),
    ("자기지칭", r"내가|제가|나는"),
    ("조언_해야지", r"해야지|해야죠|하셔야|하는 게 좋|하시는 게"),
    ("인생_담론", r"사람|인생|생각|마음|진심"),
    ("확인_의문", r"어쩌죠|그쵸|그죠|어떠신가요|어떨까요|아닌가요"),
    ("존칭_님씨", r"님|씨"),

    # ----- 일반(젊은 세대) 신호 -----
    ("웃음_ㅋㅋ", r"ㅋ{2,}"),
    ("우는_ㅠㅠ", r"[ㅠㅜ]{2,}"),
    ("초성체", r"ㄱㄱ|ㅇㅇ|ㄴㄴ|ㄹㅇ|ㅇㅋ|ㄷㄷ|ㅅㄱ|ㅈㅅ"),
    ("물음표", r"\?"),
    ("신조어_줄임말", r"실화냐|갑분|킹받|어쩔|팀플|에타|빡세|개[가-힣]|존나|점메추"),
]


FEATURE_NAMES = [name for name, _ in FEATURES]

# 정규식은 매번 컴파일하지 않고 한 번만 컴파일해서 재사용합니다.
COMPILED_PATTERNS = [re.compile(pattern) for _, pattern in FEATURES]


# ==========================================================
# 3. 등급 기준 (PRD 4.2 Nice to have)
# ==========================================================

GRADES = [
    (20, "진성 MZ", "영포티 특징이 거의 없습니다."),
    (40, "정상 범주", "무난한 말투입니다."),
    (60, "영포티 경계", "슬슬 물결(~)과 마침표를 조심하세요."),
    (80, "MZ 코스프레 40대", "젊은 척이 살짝 티납니다."),
    (101, "확진 영포티", "라떼 향이 진하게 감지되었습니다."),
]


def get_grade(score):
    """영포티 지수(0~100)를 5단계 등급으로 변환합니다."""

    for upper, name, comment in GRADES:
        if score < upper:
            return name, comment

    return GRADES[-1][1], GRADES[-1][2]


# ==========================================================
# 4. Multi-hot 벡터화 (PRD 5.3 핵심 함수)
# ==========================================================


def text_to_multihot(text, patterns=COMPILED_PATTERNS):
    """
    문장 1개를 28차원 multi-hot 벡터로 변환합니다.

    특징이 등장한 '횟수'를 그대로 값으로 사용합니다.

    예:

    "라떼는 말이야 우리 때는!! 나 때는 진짜 힘들었어"

    라떼_나때 → 3 (라떼, 우리 때, 나 때)
    느낌표    → 2
    나머지    → 0
    """

    if text is None:
        text = ""

    text = str(text)

    vector = []

    for pattern in patterns:
        count = len(pattern.findall(text))
        vector.append(count)

    return np.array(vector, dtype=float)


def transform_texts(texts):
    """여러 문장을 (샘플 수, 28) 크기의 multi-hot 행렬로 변환합니다."""

    return np.array([text_to_multihot(text) for text in texts])


# ==========================================================
# 5. 프로그램 시작
# ==========================================================

print()
print("=" * 70)
print("                        FORTY-SCAN")
print("                     영포티 말투 판독기")
print("             Multi-hot encoding + Logistic Regression")
print("=" * 70)


# ==========================================================
# 6. CSV 불러오기
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
# 7. Multi-hot 벡터 변환
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
# 8. Train / Test 분할
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
# 9. 모델 학습 및 비교 (PRD 5.3 모델 후보 3종)
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
# 10. 성능 평가 (분류 4대 지표)
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
# 11. 모델 저장
# ==========================================================

print()
print("[6/6] 모델 저장 중...")

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)

print(f"      저장 완료 : {MODEL_PATH}")


# ==========================================================
# 12. 판독 함수
# ==========================================================


def predict_forty(text):
    """
    문장 1개를 판독합니다.

    반환값:
    score        : 영포티 지수 (0~100)
    vector       : 28차원 multi-hot 벡터
    contributions: (특징명, 등장횟수, 계수, 기여도) 리스트 (기여도 절댓값 내림차순)
    """

    vector = text_to_multihot(text)

    score = model.predict_proba(vector.reshape(1, -1))[0][1] * 100

    contributions = []

    for name, count, coef in zip(FEATURE_NAMES, vector, coefficients):
        if count > 0:
            contributions.append((name, int(count), coef, count * coef))

    contributions.sort(key=lambda item: abs(item[3]), reverse=True)

    return score, vector, contributions


# ==========================================================
# 13. 대화형 판독 루프
# ==========================================================

print()
print("=" * 70)
print(" [영포티 판독기]  분석할 문장을 입력하세요.")
print(" 종료하려면 q 또는 exit 를 입력하세요.")
print("=" * 70)

while True:

    try:
        user_input = input("\n판독할 문장 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n판독기를 종료합니다.")
        break

    if user_input.lower() in ("q", "exit", "quit"):
        print("판독기를 종료합니다.")
        break

    if not user_input:
        continue

    score, vector, contributions = predict_forty(user_input)
    grade, comment = get_grade(score)

    print()
    print("-" * 70)
    print(f" 영포티 지수 : {score:.1f}%   [{grade}]")
    print(f" {comment}")
    print("-" * 70)

    if contributions:
        print(f" {'검출된 특징':<18}{'횟수':>6}{'가중치':>10}{'기여도':>10}")
        print(" " + "-" * 68)

        for name, count, coef, contribution in contributions:
            print(f" {name:<18}{count:>6}{coef:>+10.2f}{contribution:>+10.2f}")
    else:
        print(" 사전에 등록된 특징이 검출되지 않았습니다.")

    print(" " + "-" * 68)
    print(f" multi-hot 벡터 : {vector.astype(int).tolist()}")
    print("-" * 70)
