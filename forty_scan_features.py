# ==========================================================
# forty-scan
# 영포티 말투 판독기 - 특징추출 공용 모듈
#
# forty_scan_model.py 와 forty_scan_test.py 가
# 공통으로 import 하는 모듈입니다.
# 학습 시와 판독(테스트) 시 특징추출 로직이 반드시 동일해야 하므로
# 이 파일 하나로 관리합니다.
#
# 기반 문서:
# 영포티판독기_PRD_v2.0.md
#
# 특징 인코딩:
# Multi-hot encoding (등장 횟수를 그대로 값으로 사용)
#
# 0회 → 0
# 1회 → 1
# 3회 → 3
#
# ==========================================================


# ==========================================================
# 0. 라이브러리
# ==========================================================

import os
import re

import numpy as np


# ==========================================================
# 1. 기본 설정
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "forty_scan_현경_model.pkl")

# 오타 보정 킬 스위치
#
# True  : 자모 정규화 + 퍼지 매칭을 적용해 오타가 섞인 문장도 특징을 잡습니다.
# False : 오타 보정 경로를 통째로 건너뛰고 기존 정규식 결과만 사용합니다.
#
# 기존 FEATURES 정규식은 한 글자도 수정하지 않았으므로,
# 이 값을 False로 두면 오타 보정 도입 이전과 100% 동일하게 동작합니다.
USE_TYPO_CORRECTION = True


# ==========================================================
# 2. 특징 사전 (29개 차원)
# ==========================================================
#
# PRD 5.2 / 5.3 기준으로 확정한 29개 특징입니다.
# (영포티 신호 24개 + 젊은 세대 신호 5개)
#
# data/forty-scan_feature.csv 의 25개 리터럴 키워드는
# 아래 카테고리 정규식 안에 모두 흡수했습니다.
#
# 예) 미인 / 미녀 / 늘씬 / 아름다우셔서  →  외모_칭찬
#     오빠로써 / 인생선배로써 / 살아보니까 →  인생선배_훈수
#
# 앞의 24개는 영포티 신호,
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
    ("말줄임표", r"\.{2,}|…"),
    # 존댓말 자체는 세대 신호가 아니므로, 존댓말 뒤에 물결/이모티콘이 붙는
    # '영포티식 존댓말'만 잡습니다. (예: 하셨네요~ / 어떠신가요~^^)
    ("존댓말_물결조합", r"(?:요|죠|네요|세요|어요|아요)\s*[~^]|~\s*\^\^"),
    ("쉼표_남용", r",{2,}"),
    ("이모지_특수기호", r"[\U0001F300-\U0001FAFF☀-➿♥♡❤]"),

    # ----- 영포티 신호 : 어휘 -----
    ("오빠_호칭", r"오빠|옵빠|누나"),
    ("외모_칭찬", r"처자|미인|미녀|늘씬|아름다우|아름다운|예쁘|이쁘|동안|훈남|아름다워"),
    ("인연_운명", r"인연|운명|만남 시작"),
    ("인생선배_훈수", r"인생선배|오빠로써|로써|살아보니까|살다 보니|조언"),
    ("라떼_나때", r"라떼|나 때|우리 때|내가 젊|왕년"),
    ("요즘애들", r"요즘 애들|요즘 것들|요즘 젊은|요즘 친구"),
    ("젊음_강조", r"젊음|젊은 애들|젊은 남자|핫바리|젊게"),
    ("옛날_유행어", r"핫플레이스|고고씽|방가|열정|대쉬|전번|므흣|짱"),
    ("감탄사_에혀", r"에혀|어허|허허|아이고|어이쿠|허참"),

    # ----- 영포티 신호 : 문장 습관 -----
    ("격식체_습니다", r"습니다|입니다|합니다|니다"),
    ("자기지칭", r"내가|중년"),
    ("상대지칭", r"우리 이쁜|양"),
    ("조언_해야지", r"해야지|해야죠|하셔야|하는 게 좋|하시는 게"),
    ("인생_담론", r"사람|인생|생각|마음|진심"),
    ("확인_의문", r"어쩌죠|그쵸|그죠|어떠신가요|어떨까요|아닌가요"),
    ("존칭_님씨", r"님|씨"),

    # ----- 일반(젊은 세대) 신호 -----
    ("초성체", r"ㄱㄱ|ㅇㅇ|ㄴㄴ|ㄹㅇ|ㅇㅋ|ㄷㄷ|ㅅㄱ|ㅈㅅ"),
    ("신조어_줄임말", r"실화냐|갑분|킹받|어쩔|팀플|에타|빡세|개[가-힣]|존나|점메추|야르|샤갈|에바"),
]


FEATURE_NAMES = [name for name, _ in FEATURES]

# 정규식은 매번 컴파일하지 않고 한 번만 컴파일해서 재사용합니다.
COMPILED_PATTERNS = [re.compile(pattern) for _, pattern in FEATURES]


# ==========================================================
# 2.5 오타 보정 (자모 정규화 + 자모 편집거리 퍼지 매칭)
# ==========================================================
#
# 문제:
#   위 정규식은 글자가 정확히 일치해야만 잡습니다.
#   그래서 키워드 어간 안쪽에 오타가 있으면 특징이 통째로 사라집니다.
#
#   "아름따우셔서"  → 외모_칭찬 놓침 (ㄷ→ㄸ)
#   "라때는"        → 라떼_나때 놓침 (ㅔ→ㅐ)
#   "살아보닛가"    → 인생선배_훈수 놓침 (ㄲ→ㅅ+ㄱ)
#
# 왜 해밍거리가 아닌가:
#   해밍거리는 길이가 같은 문자열에만 정의됩니다.
#   그런데 한글 오타는 대부분 글자가 빠지거나 늘어나는 삽입/삭제입니다.
#   ("나 때" 3자 ↔ "나때" 2자)  → 해밍거리를 아예 계산할 수 없습니다.
#
#   또 음절 단위로 비교하면 오타의 크기가 왜곡됩니다.
#   "라떼"/"라때"는 음절로 보면 2글자 중 1글자(50%)가 다르지만,
#   실제로는 ㅔ→ㅐ 자모 하나 차이일 뿐입니다.
#
#   → 삽입/삭제를 다루는 편집거리(Levenshtein)를 자모 단위로 적용합니다.
#
# 3계층 구조:
#
#   원문 ─┬─→ [1층] 기존 정규식 findall             → regex_count
#         │
#         └─→ 자모 분해 + [0층] 자모 정규화
#                    └─→ [2층] 자모 편집거리 윈도우 → fuzzy_count
#
#              최종 count = max(regex_count, fuzzy_count)
#
#   * 정규식은 반드시 '원문'에서만 돌립니다.
#     자모 분해된 텍스트에 기존 정규식을 돌리면 ㅎ{2,} / ㅋ{2,} / 초성체
#     같은 패턴이 오작동합니다. ("하하" → "ㅎㅏㅎㅏ" 처럼 자모가 노출됨)
#     퍼지 경로는 기존 동작을 건드리지 않는 별도 채널로만 더합니다.
#
#   * max()를 쓰는 이유:
#     퍼지 매칭은 오차 0인 정확 일치도 당연히 잡으므로,
#     두 값을 더하면 같은 등장을 두 번 세게 됩니다.
#     max는 이중 계수를 구조적으로 막고 값이 정수로 유지됩니다.
#
# ==========================================================

# ----- 자모 분해용 유니코드 상수 -----
#
# 한글 음절 '가'(0xAC00) ~ '힣'(0xD7A3)은 다음 규칙으로 만들어집니다.
#
#   코드 = 0xAC00 + (초성index * 588) + (중성index * 28) + 종성index
#
# 따라서 거꾸로 나누면 초/중/종성을 복원할 수 있습니다.

HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3

CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSUNG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


# ----- [0층] 자모 정규화 테이블 -----
#
# 한글 "오타"의 상당수는 사실 발음이 비슷해서 생기는 음운 혼동입니다.
# 혼동되는 자모를 하나로 접어두면 편집거리를 재기도 전에 0이 되어버립니다.
# 임계값 판단이 필요 없으므로 오탐 위험이 사실상 없는, 가장 안전한 계층입니다.
#
#   라때  → 라떼      아름따우 → 아름다우      인셍선배 → 인생선배

JAMO_NORMALIZE = str.maketrans(
    {
        # 된소리 → 예사소리
        "ㄲ": "ㄱ",
        "ㄸ": "ㄷ",
        "ㅃ": "ㅂ",
        "ㅆ": "ㅅ",
        "ㅉ": "ㅈ",
        # 구분이 잘 안 되는 모음
        "ㅐ": "ㅔ",
        "ㅒ": "ㅖ",
        "ㅙ": "ㅚ",
        "ㅞ": "ㅚ",
        "ㅢ": "ㅣ",
    }
)


def decompose_jamo(text):
    """
    문자열을 자모 단위로 분해하고 [0층] 정규화까지 적용합니다.

    반환값:
    jamo   : 정규화된 자모 문자열
    starts : 원본 '글자'가 시작되는 jamo 인덱스 집합

    starts가 필요한 이유:
    자모를 이어 붙이면 글자 경계가 사라져서, 앞 글자의 종성부터
    매칭이 시작되는 엉뚱한 결과가 나올 수 있습니다.
    매칭 시작점을 글자 경계로 제한해 이런 헛매칭을 막습니다.

    예) "라떼" → jamo "ㄹㅏㄷㅔ", starts {0, 2}
    """

    pieces = []
    starts = set()
    position = 0

    for char in text:
        starts.add(position)

        code = ord(char)

        if HANGUL_BASE <= code <= HANGUL_LAST:
            offset = code - HANGUL_BASE

            cho = CHOSUNG[offset // 588]
            jung = JUNGSUNG[offset % 588 // 28]
            jong = JONGSUNG[offset % 28]

            # 종성이 없으면 JONGSUNG[0]인 공백이므로 붙이지 않습니다.
            piece = cho + jung + (jong if jong != " " else "")
        else:
            piece = char

        pieces.append(piece)
        position += len(piece)

    jamo = "".join(pieces).translate(JAMO_NORMALIZE)

    return jamo, starts


def bounded_levenshtein(source, target, cap):
    """
    편집거리를 계산하되, cap을 넘는 순간 포기하고 cap+1을 돌려줍니다.

    전체 거리를 끝까지 구할 필요가 없습니다.
    우리는 "cap 이하인가?"만 알면 되므로, 조기 종료로 크게 빨라집니다.
    """

    if abs(len(source) - len(target)) > cap:
        return cap + 1

    previous = list(range(len(target) + 1))

    for i, source_char in enumerate(source, start=1):
        current = [i]

        for j, target_char in enumerate(target, start=1):
            current.append(
                min(
                    previous[j] + 1,               # 삭제
                    current[j - 1] + 1,            # 삽입
                    previous[j - 1] + (source_char != target_char),  # 대체
                )
            )

        # 이 행의 최솟값이 이미 cap을 넘었다면 최종 거리도 cap을 넘습니다.
        if min(current) > cap:
            return cap + 1

        previous = current

    return previous[-1]


def error_budget(keyword_jamo):
    """
    키워드 길이에 비례한 '허용 오차 개수'를 정합니다.

    고정 유사도 임계값(예: 0.8 이상이면 같은 단어)을 쓰면 안 됩니다.
    짧은 단어에서는 전혀 다른 단어가 같은 점수를 받기 때문입니다.

        미인 / 미안  → 자모 5개, 편집거리 1  (다른 단어)
        인연 / 인원  → 자모 6개, 편집거리 1  (다른 단어)

    그래서 짧은 키워드에는 오차를 아예 허용하지 않고,
    길어질수록 오차를 늘립니다.
    """

    length = len(keyword_jamo)

    if length <= 6:      # 2음절 이하 → 오타 허용 안 함
        return 0
    if length <= 11:     # 3~4음절
        return 1
    return 2             # 5음절 이상


def _match_at(text_jamo, index, keyword_jamo, budget):
    """
    text_jamo의 index 위치에서 키워드가 오차 budget 이내로 시작하는지 봅니다.

    맞으면 매칭된 길이를, 아니면 0을 돌려줍니다.
    """

    keyword_length = len(keyword_jamo)
    text_length = len(text_jamo)

    shortest = max(1, keyword_length - budget)

    for window_length in range(shortest, keyword_length + budget + 1):

        if index + window_length > text_length:
            break

        window = text_jamo[index : index + window_length]

        if bounded_levenshtein(keyword_jamo, window, budget) <= budget:
            return window_length

    return 0


def fuzzy_count(text_jamo, starts, entries):
    """
    한 특징의 키워드들이 문장에 몇 번 '등장'하는지 셉니다.

    키워드마다 따로 세서 더하면 안 됩니다.
    한 특징 안에 비슷한 변형이 여러 개 들어 있기 때문입니다.

        외모_칭찬 = [... "아름다우", "아름다운", "아름다워" ...]

    "아름다우셔서"는 이 셋에 모두 걸리므로 따로 세면 1번 등장이 3으로 부풀려집니다.
    그래서 문장을 한 번만 훑으면서, 같은 자리에서 여러 키워드가 걸려도 1로 셉니다.

    글자 경계(starts)에서만 매칭을 시작하고,
    한 번 매칭되면 그 구간을 건너뛰어 같은 부분을 두 번 세지 않습니다.
    """

    text_length = len(text_jamo)

    count = 0
    index = 0

    while index < text_length:

        if index not in starts:
            index += 1
            continue

        longest = 0

        for keyword_jamo, budget in entries:

            # 텍스트 남은 길이가 키워드보다 짧으면 볼 것도 없습니다. (프리필터)
            if text_length - index < len(keyword_jamo) - budget:
                continue

            matched = _match_at(text_jamo, index, keyword_jamo, budget)

            # 같은 자리에서 여러 키워드가 걸리면 가장 긴 것만큼 건너뜁니다.
            if matched > longest:
                longest = matched

        if longest:
            count += 1
            index += longest   # 겹치지 않게 건너뜁니다.
        else:
            index += 1

    return count


# ----- 퍼지 매칭 대상 키워드 -----
#
# 정규식에서 자동으로 뽑지 않고 손으로 나열합니다.
# 기존 패턴에는 문자 클래스([가-힣]), 수량자({2,}), 대안(|)이 섞여 있어서
# 그대로는 퍼지 매칭 대상이 될 수 없기 때문입니다.
#
# 여기 없는 특징(물결/느낌표/이모지/ㅎㅎ/ㅋㅋ/초성체 등)은
# 기호나 표기 습관이라 오타 보정이 의미가 없고 위험하기만 합니다.
#
# 짧고 흔한 단어(내가/제가/사람/인생/님/씨/양)도 일부러 뺐습니다. 오탐의 원인입니다.
#
# 정규화 충돌로 뺀 것:
#   "로써" → 정규화하면 "로서"와 같아져 평범한 문장을 다 잡습니다.
#   "짱"   → 정규화하면 "장"과 같아져 사장/장소까지 잡습니다.

FUZZY_KEYWORDS = {
    "오빠_호칭": ["오빠", "옵빠", "누나"],
    "외모_칭찬": [
        "처자", "미인", "미녀", "늘씬", "아름다우", "아름다운", "아름다워",
        "예쁘", "이쁘", "동안", "훈남",
    ],
    "인연_운명": ["인연", "운명", "만남 시작"],
    "인생선배_훈수": ["인생선배", "오빠로써", "살아보니까", "살다 보니", "조언"],
    "라떼_나때": ["라떼", "나 때", "우리 때", "내가 젊", "왕년"],
    "요즘애들": ["요즘 애들", "요즘 것들", "요즘 젊은", "요즘 친구"],
    "젊음_강조": ["젊음", "젊은 애들", "젊은 남자", "핫바리", "젊게"],
    "옛날_유행어": ["핫플레이스", "고고씽", "방가", "열정", "대쉬", "전번", "므흣"],
    "감탄사_에혀": ["에혀", "어허", "허허", "아이고", "어이쿠", "허참"],
    "조언_해야지": ["해야지", "해야죠", "하셔야", "하는 게 좋", "하시는 게"],
    "확인_의문": ["어쩌죠", "그쵸", "그죠", "어떠신가요", "어떨까요", "아닌가요"],
    "신조어_줄임말": [
        "실화냐", "갑분", "킹받", "어쩔", "팀플", "에타", "빡세",
        "존나", "점메추", "야르", "샤갈", "에바",
    ],
}


# ----- 별칭 사전 -----
#
# 위 오차 예산 규칙은 짧은 단어의 오타를 '일부러' 막습니다. (미인/미안 오탐 방지)
# 그래서 막혔지만 실제로 필요한 변형은 여기에 리터럴로 적어 둡니다.
#
# 별칭은 항상 오차 0으로만 매칭하므로 새로운 오탐을 만들지 않습니다.
#
# 기존 FEATURES 정규식은 한 글자도 고치지 않았습니다.
# 정규식에 별칭을 끼워 넣으면 USE_TYPO_CORRECTION = False 로 되돌릴 수 없게 되어
# 킬 스위치가 반쪽이 되기 때문입니다.

ALIASES = {
    "오빠_호칭": ["옵바"],
    # "아이고"는 자모 6개라 예산이 0입니다. 흔한 변형만 리터럴로 열어 둡니다.
    "감탄사_에혀": ["아이구", "어이구"],
    # 정규식이 띄어쓰기를 요구하는 것들의 붙여쓴 형태
    "라떼_나때": ["나때", "우리때"],
    "요즘애들": ["요즘애들", "요즘것들", "요즘젊은", "요즘친구"],
    "젊음_강조": ["젊은애들", "젊은남자"],
    "인연_운명": ["만남시작"],
}


def _build_fuzzy_table():
    """
    특징 이름 → [(키워드 자모, 오차 예산), ...] 표를 미리 만들어 둡니다.

    자모 분해와 예산 계산은 문장마다 달라지지 않으므로 시작할 때 한 번만 합니다.
    """

    table = {}

    for name in FEATURE_NAMES:

        entries = []

        for keyword in FUZZY_KEYWORDS.get(name, []):
            keyword_jamo, _ = decompose_jamo(keyword)
            entries.append((keyword_jamo, error_budget(keyword_jamo)))

        # 별칭은 예산을 무조건 0으로 고정합니다.
        for alias in ALIASES.get(name, []):
            alias_jamo, _ = decompose_jamo(alias)
            entries.append((alias_jamo, 0))

        if entries:
            table[name] = entries

    return table


FUZZY_TABLE = _build_fuzzy_table()


# ==========================================================
# 3. 등급 기준 (PRD 4.2 Nice to have)
# ==========================================================

GRADES = [
    (20, "완전 청정", "영포티 바이러스 반응 0% 상태입니다."),
    (40, "정상 범주", "극히 정상적이고 담백한 대화 패턴입니다."),
    (60, "영포티 경계", "은연중에 아재 감성이 스며드는 단계입니다."),
    (80, "영포티 고위험군", "억지 유행어와 감성 마침표가 과다 사용되고 있습니다."),
    (101, "말기 영포티", "라떼 향 가득한 영포티 마스터 단계입니다."),
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


def text_to_multihot(text, patterns=COMPILED_PATTERNS, return_fuzzy_flags=False):
    """
    문장 1개를 29차원 multi-hot 벡터로 변환합니다.

    특징이 등장한 '횟수'를 그대로 값으로 사용합니다.

    예:

    "라떼는 말이야 우리 때는!! 나 때는 진짜 힘들었어"

    라떼_나때 → 3 (라떼, 우리 때, 나 때)
    느낌표    → 2
    나머지    → 0

    USE_TYPO_CORRECTION이 True면 정규식이 놓친 오타를
    자모 편집거리로 한 번 더 찾아봅니다. (섹션 2.5 참고)

    return_fuzzy_flags=True로 부르면
    '정규식은 못 잡았는데 오타 보정이 잡은' 특징 이름 집합을 함께 돌려줍니다.
    판독 결과에 (오타보정) 표시를 붙이는 데 씁니다.
    """

    if text is None:
        text = ""

    text = str(text)

    # 자모 분해는 문장당 한 번만 합니다. (키워드마다 다시 분해하면 느려집니다)
    if USE_TYPO_CORRECTION and FUZZY_TABLE:
        text_jamo, starts = decompose_jamo(text)
    else:
        text_jamo, starts = "", set()

    vector = []
    fuzzy_flags = set()

    for name, pattern in zip(FEATURE_NAMES, patterns):

        # [1층] 기존 정규식 - 반드시 '원문'에서 돌립니다.
        count = len(pattern.findall(text))

        # [2층] 오타 보정 - 별도 채널로 더합니다.
        entries = FUZZY_TABLE.get(name) if text_jamo else None

        if entries:
            fuzzy = fuzzy_count(text_jamo, starts, entries)

            if fuzzy > count:
                # 퍼지는 정확 일치도 잡으므로 더하면 이중 계수가 됩니다. max로 막습니다.
                if count == 0:
                    fuzzy_flags.add(name)
                count = fuzzy

        vector.append(count)

    vector = np.array(vector, dtype=float)

    if return_fuzzy_flags:
        return vector, fuzzy_flags

    return vector


def transform_texts(texts):
    """여러 문장을 (샘플 수, 29) 크기의 multi-hot 행렬로 변환합니다."""

    return np.array([text_to_multihot(text) for text in texts])
