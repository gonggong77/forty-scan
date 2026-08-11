# ==========================================================
# forty-scan
# 영포티 말투 판독기 - 테스트(대화형 판독) 스크립트
#
# forty_scan_현경_모델생성.py 로 저장한 모델(model/forty_scan_현경_model.pkl)을
# 불러와 문장을 입력받아 영포티 지수를 판독합니다.
#
# 특징추출(정규식 사전 + 오타 보정)은 forty_scan_features.py 를
# import 해서 씁니다. (모델 생성 스크립트와 동일한 전처리를 공유하기 위함)
#
# 기능:
# 1. 저장된 모델 불러오기
# 2. 새로운 문장 반복 판독
# 3. 검출된 특징 / 등장 횟수 / 기여도 출력
# 4. 5단계 등급 판정
#
# ==========================================================


# ==========================================================
# 0. 라이브러리
# ==========================================================

import os
import sys

import joblib

from forty_scan_features import (
    FEATURE_NAMES,
    MODEL_PATH,
    get_grade,
    text_to_multihot,
)


# 윈도우 콘솔에서 한글/이모지가 깨지지 않도록 출력 인코딩 지정
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==========================================================
# 1. 저장된 모델 불러오기
# ==========================================================

print()
print("=" * 70)
print("                        FORTY-SCAN")
print("                     영포티 말투 판독기 - 테스트")
print("=" * 70)
print()
print("모델 불러오는 중...")
print(f"      파일 경로 : {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "\n모델 파일을 찾을 수 없습니다.\n"
        f"현재 경로:\n{MODEL_PATH}\n\n"
        "forty_scan_현경_모델생성.py 를 먼저 실행해서 모델을 생성하세요."
    )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
coefficients = model.coef_[0]

print("      불러오기 완료")


# ==========================================================
# 2. 판독 함수
# ==========================================================


def predict_forty(text):
    """
    문장 1개를 판독합니다.

    반환값:
    score        : 영포티 지수 (0~100)
    vector       : 29차원 multi-hot 벡터
    contributions: (특징명, 등장횟수, 계수, 기여도, 오타보정여부) 리스트
                   (기여도 절댓값 내림차순)
    """

    vector, fuzzy_flags = text_to_multihot(text, return_fuzzy_flags=True)

    score = model.predict_proba(vector.reshape(1, -1))[0][1] * 100

    contributions = []

    for name, count, coef in zip(FEATURE_NAMES, vector, coefficients):
        if count > 0:
            contributions.append(
                (name, int(count), coef, count * coef, name in fuzzy_flags)
            )

    contributions.sort(key=lambda item: abs(item[3]), reverse=True)

    return score, vector, contributions


# ==========================================================
# 3. 대화형 판독 루프
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
        print(f" {'검출된 특징':<18}{'횟수':>6}{'가중치':>10}{'기여도':>10}   비고")
        print(" " + "-" * 68)

        for name, count, coef, contribution, is_fuzzy in contributions:
            # 정규식은 놓쳤는데 오타 보정이 잡아낸 특징을 표시해 줍니다.
            note = "(오타보정)" if is_fuzzy else ""
            print(
                f" {name:<18}{count:>6}{coef:>+10.2f}{contribution:>+10.2f}   {note}"
            )
    else:
        print(" 사전에 등록된 특징이 검출되지 않았습니다.")

    print(" " + "-" * 68)
    print(f" multi-hot 벡터 : {vector.astype(int).tolist()}")
    print("-" * 70)
