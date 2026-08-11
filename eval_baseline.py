# ==========================================================
# eval_baseline.py
# 오타 보정 도입 전/후 성능 비교 (롤백 판정용)
#
# 왜 이 스크립트가 따로 필요한가
# ------------------------------------------------------
# forty_scan_현경.py 가 출력하는 Accuracy / F1 은
# train_test_split 을 한 번만 한 결과입니다. 검증 데이터가 477개뿐이라
# 코드를 전혀 바꾸지 않고 seed만 바꿔도 F1이 이만큼 흔들립니다.
#
#     LogisticRegression F1  평균 0.7720  표준편차 0.0227
#                            범위 0.7326 ~ 0.8307   (± 0.049)
#
# 즉 "F1이 0.7543에서 0.74로 떨어졌다"는 것만으로는
# 오타 보정 탓인지 분할 운인지 구분할 수 없습니다.
#
# 그래서 여기서는 5-fold 교차검증으로,
# 오타 보정 OFF / ON 두 조건에 '똑같은 폴드'를 적용해 비교합니다.
#
# 이 스크립트는 모델 파일을 저장하지 않습니다. (읽기 전용 평가)
#
# 실행: python eval_baseline.py
# ==========================================================

import os
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

sys.stdout.reconfigure(encoding="utf-8")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(BASE_DIR, "forty_scan_현경.py")

SENTINEL = "# 5. 프로그램 시작"

RANDOM_STATE = 42

# 롤백 판정 기준 (5-fold 교차검증 F1 평균, LogisticRegression)
#
# 작업 전 실측한 기준선입니다. 이 값과 비교해 채택/축소/롤백을 정합니다.
BASELINE_F1 = 0.7696
BASELINE_F1_STD = 0.0248


def load_definitions(use_typo_correction):
    """forty_scan_현경.py 의 정의부만 실행합니다. (학습/대화 루프는 실행 안 함)"""

    with open(SOURCE_PATH, encoding="utf-8") as source_file:
        source = source_file.read()

    namespace = {"__name__": "forty_scan_defs", "__file__": SOURCE_PATH}
    exec(compile(source[: source.index(SENTINEL)], SOURCE_PATH, "exec"), namespace)

    namespace["USE_TYPO_CORRECTION"] = use_typo_correction

    return namespace


def load_dataframe(namespace):
    """forty_scan_현경.py 와 똑같은 순서로 데이터를 정리합니다."""

    df = pd.read_csv(namespace["DATA_PATH"], encoding="utf-8")

    polite_path = namespace["POLITE_PATH"]

    if os.path.exists(polite_path):
        df = pd.concat(
            [df, pd.read_csv(polite_path, encoding="utf-8")], ignore_index=True
        )

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    return df


def make_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def evaluate(X, y, folds):
    """모델 3종을 교차검증해 {이름: (acc평균, acc표준편차, f1평균, f1표준편차)}를 돌려줍니다."""

    scores = {}

    for name, model in make_models().items():
        accuracy = cross_val_score(model, X, y, cv=folds, scoring="accuracy")
        f1 = cross_val_score(model, X, y, cv=folds, scoring="f1")

        scores[name] = (accuracy.mean(), accuracy.std(), f1.mean(), f1.std())

    return scores


def main():
    print()
    print("=" * 86)
    print("            오타 보정 성능 비교  (5-fold 교차검증, 롤백 판정용)")
    print("=" * 86)

    print()
    print("데이터 준비 중...")

    off = load_definitions(use_typo_correction=False)
    on = load_definitions(use_typo_correction=True)

    df = load_dataframe(off)
    texts = df["text"].tolist()
    y = df["label"].to_numpy()

    print(f"  {len(df)}행 (일반 {int((y == 0).sum())}, 영포티 {int((y == 1).sum())})")

    print("벡터 변환 중... (보정 OFF)")
    X_off = off["transform_texts"](texts)

    print("벡터 변환 중... (보정 ON)")
    X_on = on["transform_texts"](texts)

    # 오타 보정이 실제로 특징을 더 잡았는지 먼저 확인합니다.
    changed = int((X_on != X_off).any(axis=1).sum())
    coverage_off = (X_off.sum(axis=1) > 0).mean() * 100
    coverage_on = (X_on.sum(axis=1) > 0).mean() * 100

    print()
    print(f"  벡터가 달라진 문장 : {changed}개 ({changed / len(df) * 100:.1f}%)")
    print(f"  특징 1개 이상 검출 : {coverage_off:.1f}%  →  {coverage_on:.1f}%")

    # 두 조건에 동일한 폴드를 적용해야 비교가 성립합니다.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print()
    print("교차검증 중...")

    scores_off = evaluate(X_off, y, folds)
    scores_on = evaluate(X_on, y, folds)

    print()
    print("-" * 86)
    print(
        f" {'모델':<22}{'보정OFF Acc':>16}{'보정ON Acc':>16}"
        f"{'보정OFF F1':>16}{'보정ON F1':>16}"
    )
    print("-" * 86)

    for name in scores_off:
        acc_off, acc_off_std, f1_off, f1_off_std = scores_off[name]
        acc_on, acc_on_std, f1_on, f1_on_std = scores_on[name]

        print(
            f" {name:<22}"
            f"{f'{acc_off:.4f}±{acc_off_std:.4f}':>16}"
            f"{f'{acc_on:.4f}±{acc_on_std:.4f}':>16}"
            f"{f'{f1_off:.4f}±{f1_off_std:.4f}':>16}"
            f"{f'{f1_on:.4f}±{f1_on_std:.4f}':>16}"
        )

    print("-" * 86)

    # ----- 롤백 판정 -----
    #
    # 판정은 LogisticRegression 기준입니다. 최종 채택 모델이기 때문입니다.

    f1_on = scores_on["LogisticRegression"][2]
    f1_off = scores_off["LogisticRegression"][2]

    delta = f1_on - f1_off

    print()
    print("=" * 86)
    print(" 롤백 판정  (LogisticRegression, 5-fold F1 평균)")
    print("-" * 86)
    print(f"   작업 전 기준선 : {BASELINE_F1:.4f} (± {BASELINE_F1_STD:.4f})")
    print(f"   보정 OFF       : {f1_off:.4f}")
    print(f"   보정 ON        : {f1_on:.4f}   ({delta:+.4f})")
    print("-" * 86)

    if f1_on >= f1_off:
        print("   판정 : 채택. 성능이 유지 또는 개선되었습니다.")
        exit_code = 0
    elif f1_on >= f1_off - BASELINE_F1_STD:
        print("   판정 : 조건부 채택. 하락폭이 1σ 이내라 노이즈 범위입니다.")
        print("          test_typo.py 의 오탐이 0인지 반드시 함께 확인하세요.")
        exit_code = 0
    elif f1_on >= f1_off - 2 * BASELINE_F1_STD:
        print("   판정 : 단계적 축소 필요. 1σ 초과로 하락했습니다.")
        print("          1) error_budget 을 0/1/2 → 0/1/1 로 조입니다.")
        print("          2) FUZZY_KEYWORDS 에서 짧고 흔한 키워드를 뺍니다.")
        print("          3) 그래도 안 되면 퍼지를 끄고 자모 정규화만 남깁니다.")
        exit_code = 1
    else:
        print("   판정 : 즉시 전체 롤백. 2σ 초과로 하락했습니다.")
        print("          forty_scan_현경.py 의 USE_TYPO_CORRECTION 을 False 로 두세요.")
        exit_code = 1

    print("=" * 86)
    print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
