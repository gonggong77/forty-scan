
# ==========================================================
# forty-scan
# 영포티 말투 판독기
#
# 모델:
# Logistic Regression
#
# 특징:
# 사용자가 지정한 25개 특징
#
# 특징 인코딩:
# Weighted Multi-hot
#
# 0회      → 0
# 1회      → 1
# 2회      → 2
# 3회      → 3
# 4회 이상 → 4
#
# 기능:
# 1. CSV 불러오기
# 2. Train / Test 분할
# 3. 25개 특징 추출
# 4. Weighted Multi-hot 변환
# 5. Logistic Regression 학습
# 6. Accuracy / Precision / Recall / F1 평가
# 7. 특징별 Logistic Regression 계수 출력
# 8. 새로운 문장 반복 판독
# 9. 검출된 특징 출력
# 10. 특징별 등장 횟수 출력
# 11. 특징별 가중치 출력
# 12. 각 특징이 영포티 확률에 미치는 방향 표시
# 13. 5단계 판정
#
# ==========================================================


# ==========================================================
# 0. 라이브러리
# ==========================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# 1. 기본 설정
# ==========================================================

CSV_PATH = r"C:\forty-scan\data\forty-scan_data.csv"

MODEL_DIR = r"C:\forty-scan\model"


# ==========================================================
# 2. Weighted Multi-hot 최대 가중치
# ==========================================================
#
# 등장 횟수에 따라
#
# 0회      → 0
# 1회      → 1
# 2회      → 2
# 3회      → 3
# 4회 이상 → 4
#
# 로 변환합니다.
#
# ==========================================================

MAX_WEIGHT = 4


# ==========================================================
# 3. 사용할 25개 특징
# ==========================================================

FEATURES = [
    "♥",
    "♡",
    "넘",
    "인연",
    "미인",
    "😎",
    "아름다우셔서",
    "만남 시작",
    "미녀",
    "젊음이",
    "음요",
    "오빠로써",
    "인생선배로써",
    "전번",
    "살아보니까",
    "대쉬",
    "어쩌죠.",
    "에혀~~",
    "젊은 핫바리",
    ",,",
    "젊은 애들보단",
    "젊은 남자보단",
    "옵빠가",
    "늘씬",
    "핫플레이스"
]


# ==========================================================
# 4. 프로그램 시작
# ==========================================================

print()
print("=" * 70)
print("                 FORTY-SCAN")
print("              영포티 말투 판독기")
print("=" * 70)


# ==========================================================
# 5. 특징 카운트 함수
# ==========================================================

def get_feature_counts(text):

    """
    입력 문장에서 25개 특징의 실제 등장 횟수를 계산합니다.

    예:

    "오빠로써 미인이시네요 오빠로써"

    결과:

    오빠로써 → 2
    미인     → 1
    나머지   → 0
    """

    if text is None:

        text = ""

    text = str(text)

    counts = {}

    for feature in FEATURES:

        counts[feature] = text.count(feature)

    return counts


# ==========================================================
# 6. Weighted Multi-hot 변환 함수
# ==========================================================

def get_weighted_vector(text):

    """
    특징 등장 횟수를 0~4의 가중치로 변환합니다.

    예:

    등장 0회 → 0
    등장 1회 → 1
    등장 2회 → 2
    등장 3회 → 3
    등장 4회 → 4
    등장 5회 → 4
    등장 10회 → 4

    즉 4를 최대 가중치로 사용합니다.
    """

    counts = get_feature_counts(text)

    vector = []

    for feature in FEATURES:

        count = counts[feature]

        weight = min(
            count,
            MAX_WEIGHT
        )

        vector.append(weight)

    return np.array(
        vector,
        dtype=float
    )


# ==========================================================
# 7. 전체 텍스트 → Weighted Multi-hot
# ==========================================================

def transform_texts(texts):

    """
    여러 문장을 25개 특징의
    Weighted Multi-hot 행렬로 변환합니다.
    """

    vectors = []

    for text in texts:

        vector = get_weighted_vector(text)

        vectors.append(vector)

    return np.array(vectors)


# ==========================================================
# 8. CSV 파일 확인
# ==========================================================

print()
print("[1/7] CSV 데이터 불러오는 중...")
print()

print(
    f"파일 경로 : {CSV_PATH}"
)


if not os.path.exists(CSV_PATH):

    raise FileNotFoundError(

        "\nCSV 파일을 찾을 수 없습니다.\n"
        f"현재 경로:\n{CSV_PATH}\n\n"
        "CSV_PATH를 실제 파일 위치로 수정하세요."

    )


# ==========================================================
# 9. CSV 읽기
# ==========================================================

try:

    df = pd.read_csv(

        CSV_PATH,

        encoding="utf-8-sig"

    )

except UnicodeDecodeError:

    df = pd.read_csv(

        CSV_PATH,

        encoding="cp949"

    )


# ==========================================================
# 10. CSV 구조 확인
# ==========================================================

print()
print("-" * 70)
print("CSV 확인")
print("-" * 70)

print(
    f"컬럼 : {list(df.columns)}"
)

print(
    f"행 개수 : {len(df)}"
)


# ==========================================================
# 11. text / label 컬럼 확인
# ==========================================================

if "text" not in df.columns:

    raise ValueError(

        "\nCSV에 'text' 컬럼이 없습니다.\n"
        f"현재 컬럼: {list(df.columns)}"

    )


if "label" not in df.columns:

    raise ValueError(

        "\nCSV에 'label' 컬럼이 없습니다.\n"
        f"현재 컬럼: {list(df.columns)}"

    )


# ==========================================================
# 12. 데이터 전처리
# ==========================================================

print()
print("[2/7] 데이터 전처리 중...")


# text 결측치 처리

df["text"] = df["text"].fillna("").astype(str)


# label 숫자 변환

df["label"] = pd.to_numeric(

    df["label"],

    errors="coerce"

)


# label 없는 행 제거

df = df.dropna(

    subset=["label"]

).copy()


# label 정수 변환

df["label"] = df["label"].astype(int)


# ==========================================================
# 13. label 검사
# ==========================================================

invalid_labels = sorted(

    set(df["label"].unique()) - {0, 1}

)


if invalid_labels:

    raise ValueError(

        "\n0 또는 1이 아닌 라벨이 발견되었습니다.\n"
        f"잘못된 라벨: {invalid_labels}"

    )


# ==========================================================
# 14. 빈 문장 제거
# ==========================================================

df = df[

    df["text"].str.strip() != ""

].copy()


# ==========================================================
# 15. 데이터 리스트
# ==========================================================

texts = df["text"].tolist()

labels = df["label"].tolist()


# ==========================================================
# 16. 데이터 확인
# ==========================================================

print()
print("-" * 70)
print("최종 데이터")
print("-" * 70)

print(
    f"전체 데이터 : {len(texts)}개"
)

print(
    f"일반인 (0)  : {labels.count(0)}개"
)

print(
    f"영포티 (1)  : {labels.count(1)}개"
)


if len(texts) == 0:

    raise ValueError(

        "사용 가능한 데이터가 없습니다."

    )


if labels.count(0) == 0:

    raise ValueError(

        "일반인(0) 데이터가 없습니다."

    )


if labels.count(1) == 0:

    raise ValueError(

        "영포티(1) 데이터가 없습니다."

    )


# ==========================================================
# 17. Train / Test 분할
# ==========================================================

print()
print("[3/7] Train / Test 데이터 분할 중...")


X_train_raw, X_test_raw, y_train, y_test = train_test_split(

    texts,

    labels,

    test_size=0.2,

    random_state=42,

    stratify=labels

)


# ==========================================================
# 18. 분할 결과
# ==========================================================

print()
print("-" * 70)
print("Train / Test 분할 결과")
print("-" * 70)

print(
    f"전체 데이터 : {len(texts)}개"
)

print(
    f"학습 데이터 : {len(X_train_raw)}개"
)

print(
    f"테스트 데이터 : {len(X_test_raw)}개"
)

print()

print("[학습 데이터]")

print(
    f"일반인(0) : {y_train.count(0)}개"
)

print(
    f"영포티(1) : {y_train.count(1)}개"
)

print()

print("[테스트 데이터]")

print(
    f"일반인(0) : {y_test.count(0)}개"
)

print(
    f"영포티(1) : {y_test.count(1)}개"
)


# ==========================================================
# 19. Weighted Multi-hot 변환
# ==========================================================

print()
print("[4/7] Weighted Multi-hot 변환 중...")


X_train = transform_texts(

    X_train_raw

)


X_test = transform_texts(

    X_test_raw

)


# ==========================================================
# 20. 특징 확인
# ==========================================================

print()
print("-" * 70)
print("특징 추출 결과")
print("-" * 70)

print(
    f"특징 개수 : {len(FEATURES)}개"
)

print(
    f"최대 가중치 : {MAX_WEIGHT}"
)

print(
    f"X_train 크기 : {X_train.shape}"
)

print(
    f"X_test 크기  : {X_test.shape}"
)


print()
print("Weighted Multi-hot 방식")
print(
    "0회 → 0"
)
print(
    "1회 → 1"
)
print(
    "2회 → 2"
)
print(
    "3회 → 3"
)
print(
    "4회 이상 → 4"
)


# ==========================================================
# 21. Logistic Regression
# ==========================================================

print()
print("[5/7] Logistic Regression 학습 중...")


model = LogisticRegression(

    max_iter=2000,

    random_state=42

)


# ==========================================================
# 22. 모델 학습
# ==========================================================

model.fit(

    X_train,

    y_train

)


print("학습 완료!")


# ==========================================================
# 23. 테스트 데이터 예측
# ==========================================================

y_pred = model.predict(

    X_test

)


# ==========================================================
# 24. 성능 평가
# ==========================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


precision = precision_score(

    y_test,

    y_pred,

    zero_division=0

)


recall = recall_score(

    y_test,

    y_pred,

    zero_division=0

)


f1 = f1_score(

    y_test,

    y_pred,

    zero_division=0

)


# ==========================================================
# 25. 성능 출력
# ==========================================================

print()
print("[6/7] 모델 성능 평가")

print()
print("=" * 70)
print("                 MODEL PERFORMANCE")
print("=" * 70)

print()

print(
    f"Accuracy  : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-Score  : "
    f"{f1 * 100:.2f}%"
)


# ==========================================================
# 26. Classification Report
# ==========================================================

print()
print("-" * 70)
print("Classification Report")
print("-" * 70)

print(

    classification_report(

        y_test,

        y_pred,

        target_names=[

            "일반인(0)",

            "영포티(1)"

        ],

        zero_division=0

    )

)


# ==========================================================
# 27. Confusion Matrix
# ==========================================================

cm = confusion_matrix(

    y_test,

    y_pred

)


print()
print("-" * 70)
print("Confusion Matrix")
print("-" * 70)

print()

print("                 예측")

print("              일반인   영포티")

print(

    f"실제 일반인   "
    f"{cm[0][0]:6d}   "
    f"{cm[0][1]:6d}"

)

print(

    f"실제 영포티   "
    f"{cm[1][0]:6d}   "
    f"{cm[1][1]:6d}"

)


# ==========================================================
# 28. 특징별 Logistic Regression 계수
# ==========================================================

print()
print("=" * 70)
print("       25개 특징별 Logistic Regression 계수")
print("=" * 70)

print()

print(
    "계수가 +이면 → 영포티 확률을 높이는 방향"
)

print(
    "계수가 -이면 → 영포티 확률을 낮추는 방향"
)

print()

feature_coefficients = pd.DataFrame({

    "feature": FEATURES,

    "coefficient": model.coef_[0]

})


feature_coefficients = feature_coefficients.sort_values(

    by="coefficient",

    ascending=False

)


for _, row in feature_coefficients.iterrows():

    feature = row["feature"]

    coefficient = row["coefficient"]


    if coefficient > 0:

        direction = "영포티 ↑"

    elif coefficient < 0:

        direction = "영포티 ↓"

    else:

        direction = "영향 거의 없음"


    print(

        f"{feature:15s} : "
        f"{coefficient:+.4f}   "
        f"{direction}"

    )


# ==========================================================
# 29. 모델 저장
# ==========================================================

print()
print("[7/7] 모델 저장 중...")


os.makedirs(

    MODEL_DIR,

    exist_ok=True

)


model_path = os.path.join(

    MODEL_DIR,

    "forty_scan_weighted_logistic_model.pkl"

)


joblib.dump(

    model,

    model_path

)


print()
print("모델 저장 완료")

print(
    f"모델 : {model_path}"
)


# ==========================================================
# 30. 새로운 문장 판독 함수
# ==========================================================

def predict_text(text):

    """
    새로운 문장을 판독합니다.

    Weighted Multi-hot:

    0회      → 0
    1회      → 1
    2회      → 2
    3회      → 3
    4회 이상 → 4

    특징의 실제 등장 횟수와
    모델에 들어가는 가중치를 함께 보여줍니다.
    """


    # ======================================================
    # 1. 특징 등장 횟수
    # ======================================================

    feature_counts = get_feature_counts(text)


    # ======================================================
    # 2. Weighted Multi-hot 벡터
    # ======================================================

    feature_vector = get_weighted_vector(text)


    # ======================================================
    # 3. 특징 분석
    # ======================================================

    detected_features = []


    for index, feature in enumerate(FEATURES):

        count = feature_counts[feature]

        weight = int(feature_vector[index])


        if count > 0:

            coefficient = model.coef_[0][index]

            detected_features.append(

                (
                    feature,
                    count,
                    weight,
                    coefficient
                )

            )


    # ======================================================
    # 4. 특징 분석 출력
    # ======================================================

    print()
    print("=" * 70)
    print("                    특징 분석")
    print("=" * 70)

    print()


    if len(detected_features) == 0:

        print(
            "⚠ 지정된 25개 특징은 발견되지 않았습니다."
        )

        print()

        print(
            "Weighted Multi-hot 벡터:"
        )

        print(
            "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
            "0, 0, 0, 0, 0]"
        )


    else:

        print(

            f"총 {len(detected_features)}개의 "
            f"특징이 발견되었습니다."

        )

        print()

        print(

            "특징              횟수     가중치     계수       영향"

        )

        print(

            "-" * 70

        )


        for feature, count, weight, coefficient in detected_features:

            if coefficient > 0:

                direction = "영포티 ↑"

            elif coefficient < 0:

                direction = "영포티 ↓"

            else:

                direction = "영향 거의 없음"


            print(

                f"{feature:15s} "
                f"{count:5d}회   "
                f"{weight:5d}      "
                f"{coefficient:+.4f}   "
                f"{direction}"

            )


    # ======================================================
    # 5. Weighted Multi-hot 벡터 출력
    # ======================================================

    print()

    print("-" * 70)

    print("Weighted Multi-hot 벡터")

    print("-" * 70)

    print()

    print(

        feature_vector.astype(int).tolist()

    )


    # ======================================================
    # 6. 특징이 하나도 없는 경우
    # ======================================================

    total_feature_count = sum(

        feature_counts.values()

    )


    if total_feature_count == 0:

        # --------------------------------------------------
        # 모든 특징이 0이면
        # Logistic Regression의 입력도 전부 0입니다.
        #
        # 이 경우 모델의 intercept만 사용되므로
        # 모든 "특징 없음" 문장이 동일한 확률을
        # 갖게 됩니다.
        #
        # 이를 방지하기 위해 별도 기준을 사용합니다.
        #
        # --------------------------------------------------

        forty_probability = 0.05

        print()

        print(
            "⚠ 특징이 하나도 없으므로 "
            "영포티 확률을 5%로 보정합니다."
        )


    else:

        # --------------------------------------------------
        # Logistic Regression 확률 계산
        # --------------------------------------------------

        probability = model.predict_proba(

            feature_vector.reshape(1, -1)

        )[0]


        # label 1 = 영포티

        forty_probability = probability[1]


    # ======================================================
    # 7. 5단계 판정
    # ======================================================

    if forty_probability < 0.20:

        level = 1

        result = "MZ"


    elif forty_probability < 0.40:

        level = 2

        result = "영포티 경계선"


    elif forty_probability < 0.60:

        level = 3

        result = "진성 영포티"


    elif forty_probability < 0.80:

        level = 4

        result = "위험 영포티"


    else:

        level = 5

        result = "재사회화 필요"


    # ======================================================
    # 8. 결과 반환
    # ======================================================

    return (

        result,

        level,

        forty_probability

    )


# ==========================================================
# 31. 판독 모드 시작
# ==========================================================

print()
print()
print("=" * 70)
print("              FORTY-SCAN 준비 완료")
print("=" * 70)

print()

print("특징 가중치 기준")

print("-" * 70)

print(
    "  0회      → 가중치 0"
)

print(
    "  1회      → 가중치 1"
)

print(
    "  2회      → 가중치 2"
)

print(
    "  3회      → 가중치 3"
)

print(
    "  4회 이상 → 가중치 4"
)

print("-" * 70)

print()

print("판정 기준")

print("-" * 70)

print(
    "  0% 이상 ~ 20% 미만   → "
    "1단계 : MZ"
)

print(
    " 20% 이상 ~ 40% 미만   → "
    "2단계 : 영포티 경계선"
)

print(
    " 40% 이상 ~ 60% 미만   → "
    "3단계 : 진성 영포티"
)

print(
    " 60% 이상 ~ 80% 미만   → "
    "4단계 : 위험 영포티"
)

print(
    " 80% 이상 ~ 100%       → "
    "5단계 : 재사회화 필요"
)

print("-" * 70)

print()

print(
    "여러 문장을 계속 입력할 수 있습니다."
)

print(
    "빈 Enter → 프로그램 종료"
)

print(
    "exit / quit / 종료 → 프로그램 종료"
)


# ==========================================================
# 32. 반복 판독
# ==========================================================

while True:

    print()
    print("=" * 70)


    user_text = input(

        "판독할 카톡 문장\n"
        "> "

    )


    # ------------------------------------------------------
    # 빈 Enter → 종료
    # ------------------------------------------------------

    if not user_text.strip():

        print()

        print(
            "프로그램을 종료합니다."
        )

        break


    # ------------------------------------------------------
    # 종료 명령어
    # ------------------------------------------------------

    if user_text.strip().lower() in [

        "exit",
        "quit",
        "종료"

    ]:

        print()

        print(
            "프로그램을 종료합니다."
        )

        break


    # ------------------------------------------------------
    # 판독 실행
    # ------------------------------------------------------

    result, level, probability = predict_text(

        user_text

    )


    # ======================================================
    # 최종 결과 출력
    # ======================================================

    print()
    print("=" * 70)
    print("                    최종 판독")
    print("=" * 70)

    print()

    print(
        f"입력 문장   : {user_text}"
    )

    print()

    print(
        f"영포티 확률 : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"등급        : "
        f"{level}단계"
    )

    print(
        f"판정        : "
        f"{result}"
    )

    print()

    print("=" * 70)