# ==========================================================
# test_typo.py
# 오타 보정 테스트셋
#
# forty_scan_현경.py 의 오타 보정(자모 정규화 + 자모 편집거리)이
#
#   1) 진짜 오타를 잡는지          (양성 케이스)
#   2) 다른 단어를 안 잡는지       (음성 케이스 / 회귀 방지)
#
# 두 가지를 한 번에 확인합니다.
#
# 음성 케이스가 더 중요합니다.
# "미인"과 "미안"은 자모 편집거리가 1로 진짜 오타와 구분이 안 되기 때문에,
# 오차 예산 규칙이 이걸 제대로 막고 있는지 항상 확인해야 합니다.
#
# 실행: python test_typo.py
# ==========================================================

import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")


# ==========================================================
# forty_scan_현경.py 의 '정의부'만 불러오기
#
# 이 스크립트는 위에서 아래로 쭉 실행되는 형태라 그냥 import 하면
# 학습과 대화형 루프까지 전부 돌아갑니다.
# 그래서 "5. 프로그램 시작" 직전까지만 실행해 함수와 사전만 가져옵니다.
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join("__pycache__\\save\\forty_scan_현경.py")

SENTINEL = "# 5. 프로그램 시작"


def load_definitions(use_typo_correction=True):
    """오타 보정 ON/OFF 상태로 정의부만 실행하고 네임스페이스를 돌려줍니다."""

    with open(SOURCE_PATH, encoding="utf-8") as source_file:
        source = source_file.read()

    cut = source.index(SENTINEL)

    namespace = {"__name__": "forty_scan_defs", "__file__": SOURCE_PATH}
    exec(compile(source[:cut], SOURCE_PATH, "exec"), namespace)

    # 플래그를 바꾸면 FUZZY_TABLE 은 그대로여도 벡터화 경로가 달라집니다.
    namespace["USE_TYPO_CORRECTION"] = use_typo_correction

    return namespace


# ==========================================================
# 테스트 케이스
#
# (문장, 기대하는 특징, 검출되어야 하는가)
# ==========================================================

POSITIVE_CASES = [
    # 어간 안쪽 오타 - 정규식이 놓치던 것들
    ("아름따우셔서 어쩌죠", "외모_칭찬", "ㄷ→ㄸ"),
    ("옵바가 말해줄게", "오빠_호칭", "ㅃ→ㅂ"),
    ("라때는 말이야", "라떼_나때", "ㅔ→ㅐ"),
    ("나때는 다 그랬어", "라떼_나때", "띄어쓰기 없음"),
    ("우리때가 좋았지", "라떼_나때", "띄어쓰기 없음"),
    ("인셍선배로서 한마디 할게", "인생선배_훈수", "ㅐ/ㅔ 혼동"),
    ("살아보닛가 알겠더라", "인생선배_훈수", "ㄲ→ㅅㄱ"),
    ("핫플레이수 가자", "옛날_유행어", "끝 글자 오타"),
    ("요즘애들은 몰라", "요즘애들", "띄어쓰기 없음"),
    ("젊은애들보다 낫지", "젊음_강조", "띄어쓰기 없음"),
    ("어떠신가여", "확인_의문", "어미 오타"),
    ("아이구 참 좋네", "감탄사_에혀", "ㅗ→ㅜ"),
    # 원래도 잡히던 것 (회귀 방지)
    ("아름다우셔서 반했어요", "외모_칭찬", "원본"),
    ("아름다우서서 반했어요", "외모_칭찬", "어미 오타(원래도 잡힘)"),
    ("라떼는 말이야", "라떼_나때", "원본"),
]

NEGATIVE_CASES = [
    # 오타가 아니라 '완전히 다른 단어' - 절대 잡히면 안 됨
    ("미안해요 오늘 늦었어요", "외모_칭찬", "미인 아님"),
    ("인원이 부족합니다", "인연_운명", "인연 아님"),
    ("동양 문화에 관심이 많아요", "외모_칭찬", "동안 아님"),
    ("조인 요청을 보냈습니다", "인생선배_훈수", "조언 아님"),
    ("사장님이 부르십니다", "옛날_유행어", "짱 아님"),
    ("학생으로서 할 일을 합니다", "인생선배_훈수", "로써 아님"),
    ("장소를 예약했어요", "옛날_유행어", "짱 아님"),
    ("오늘 회의 인원 확인 부탁드립니다", "인연_운명", "인연 아님"),
    ("미안하지만 다시 보내주세요", "외모_칭찬", "미인 아님"),
    ("운영 방식을 바꿉시다", "인연_운명", "운명 아님"),
]


def detect(namespace, text, feature_name):
    """해당 문장에서 그 특징이 몇 번 잡혔는지 돌려줍니다."""

    vector = namespace["text_to_multihot"](text)
    index = namespace["FEATURE_NAMES"].index(feature_name)

    return int(vector[index])


def main():
    print()
    print("=" * 78)
    print("                       오타 보정 테스트셋")
    print("=" * 78)

    off = load_definitions(use_typo_correction=False)
    on = load_definitions(use_typo_correction=True)

    # ----- 양성 케이스 -----

    print()
    print("[양성] 오타가 섞여도 특징이 잡혀야 하는 문장")
    print("-" * 78)
    print(f" {'문장':<26}{'기대 특징':<16}{'오타 유형':<18}{'보정OFF':>8}{'보정ON':>8}")
    print("-" * 78)

    gained = 0
    positive_hit_off = 0
    positive_hit_on = 0

    for text, feature, kind in POSITIVE_CASES:
        before = detect(off, text, feature)
        after = detect(on, text, feature)

        positive_hit_off += before > 0
        positive_hit_on += after > 0
        gained += (before == 0) and (after > 0)

        mark_before = "검출" if before else "놓침"
        mark_after = "검출" if after else "놓침"

        print(f" {text:<26}{feature:<16}{kind:<18}{mark_before:>8}{mark_after:>8}")

    print("-" * 78)
    print(
        f" 검출률 : 보정OFF {positive_hit_off}/{len(POSITIVE_CASES)}"
        f"  →  보정ON {positive_hit_on}/{len(POSITIVE_CASES)}"
        f"   (새로 잡은 오타 {gained}개)"
    )

    # ----- 음성 케이스 -----

    print()
    print("[음성] 오타가 아니라 다른 단어이므로 잡히면 안 되는 문장")
    print("-" * 78)
    print(f" {'문장':<34}{'잡히면 안 되는 특징':<18}{'보정OFF':>8}{'보정ON':>8}")
    print("-" * 78)

    false_positive_off = 0
    false_positive_on = 0

    for text, feature, _ in NEGATIVE_CASES:
        before = detect(off, text, feature)
        after = detect(on, text, feature)

        false_positive_off += before > 0
        false_positive_on += after > 0

        mark_before = "오탐!" if before else "정상"
        mark_after = "오탐!" if after else "정상"

        print(f" {text:<34}{feature:<18}{mark_before:>8}{mark_after:>8}")

    print("-" * 78)
    print(
        f" 오탐 건수 : 보정OFF {false_positive_off}건  →  보정ON {false_positive_on}건"
    )

    # ----- 판정 -----

    print()
    print("=" * 78)

    new_false_positive = false_positive_on - false_positive_off

    if new_false_positive > 0:
        print(f" 실패 : 오타 보정 때문에 오탐이 {new_false_positive}건 늘었습니다.")
        print("        오차 예산을 줄이거나 FUZZY_KEYWORDS에서 짧은 키워드를 빼세요.")
    elif gained == 0:
        print(" 경고 : 새로 잡은 오타가 없습니다. 보정이 실제로 동작하는지 확인하세요.")
    else:
        print(f" 통과 : 오타 {gained}개를 새로 잡았고, 새로 생긴 오탐은 없습니다.")

    print("=" * 78)
    print()

    return 1 if new_false_positive > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
