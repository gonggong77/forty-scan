# ==========================================================
# forty-scan - 카톡 대화 캡처 이미지 → 메시지 목록
#
# 이미지를 Claude 비전 API 로 읽어
#
#   [{"speaker": "한병준", "message": "라떼는 말이야~^^"}, ...]
#
# 형태로 만듭니다. 이 목록을 forty_web.build_results() 에 그대로
# 넘기면 TXT 업로드와 완전히 같은 채점 경로를 탑니다.
#
# 파일명 주의: forty_scan_*.py 로 지으면 .dockerignore /
# .gcloudignore 의 "학습/CLI 스크립트 제외" 패턴에 걸려
# 배포 이미지에서 조용히 빠집니다. 반드시 forty_ocr.py 로 유지하세요.
#
# 채점 모델이 물결(~), ^^, ㅋㅋㅋ, 초성체 같은 특수문자 정규식이라
# ([forty_scan_features.py]) OCR 이 이런 문자를 "정리"해버리면
# 점수가 조용히 틀려집니다. 그래서 이 파일의 프롬프트는
# 정확도보다 "원문 그대로 베끼기"를 최우선으로 둡니다.
# ==========================================================

import base64
import json
import os

import anthropic

# ==========================================================
# 0. 설정
#
# 모델을 상수로 분리해 둡니다. claude-opus-5 로 한 번 올려봤으나
# Sonnet 5 대비 정확도 차이가 크지 않아 다시 Sonnet 5 로 되돌림
# (2026-08-13). 다시 실험할 때도 이 상수 하나만 바꾸면 됩니다.
#
# 올리든 내리든, 반드시 TXT 업로드 결과와 점수를 교차 검증한 뒤에
# 바꾸세요. (계획서 "검증 4번")
# ==========================================================

OCR_MODEL = "claude-sonnet-5"

# 서버 부팅 시 한 번만 읽습니다. 없으면 기능이 꺼진 채로 서버는
# 정상 기동합니다 (forty_web.py 의 ANTHROPIC_API_KEY 읽기와 함께
# "이 저장소에서 os.environ 을 읽는 곳" 이 여기와 forty_web.py
# 두 곳이 됩니다).
ANTHROPIC_API_KEY = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()

OCR_ENABLED = bool(ANTHROPIC_API_KEY)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if OCR_ENABLED else None

# 업로드 제한 (forty_web.py 의 검증에서도 이 값을 그대로 씁니다)
MAX_IMAGES = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB

ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
}


# ==========================================================
# 1. 구조화 출력 스키마
# ==========================================================

MESSAGES_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["speaker", "message"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["messages"],
    "additionalProperties": False,
}


# ==========================================================
# 2. 프롬프트
#
# 이 기능의 성패가 여기 달려 있습니다. 채점 정규식이 반응하는
# 특수문자(~, ^^, ㅋㅋㅋ, ㅠㅠ)를 OCR 이 "교정"하면 점수가 조용히
# 틀려지므로, 정확도보다 원문 보존을 반복해서 강조합니다.
# ==========================================================

EXTRACTION_PROMPT = """카카오톡 대화 캡처 화면입니다. 화면에 보이는 대화 내용을 그대로 옮겨 적어주세요.

반드시 지켜야 할 규칙:

1. 원문 그대로 옮길 것. 오타, 띄어쓰기, 반복 문자(ㅋㅋㅋㅋ, ~~~, ㅠㅠㅠ), 이모티콘(^^, ㅠㅠ), 이모지를
   절대 교정하거나 정규화하지 마세요. 맞춤법을 고치지 마세요. 화면에 보이는 그대로가 정답입니다.

2. 화자 구분: 화면 오른쪽에 정렬된 말풍선은 화자를 "나"로 표시하세요.
   왼쪽에 정렬된 말풍선은 그 위쪽이나 프로필 옆에 적힌 이름을 화자로 쓰세요.
   이름을 찾을 수 없으면 "상대방"으로 표시하세요.

3. 다음은 메시지가 아니므로 제외하세요: 시각 표시(예: 오후 3:20), 안읽음 숫자, 날짜 구분선,
   시스템 메시지(예: "OOO님이 들어왔습니다"), 답장 인용 블록(다른 메시지를 인용한 부분).

4. 말풍선 하나 = 메시지 하나입니다. 한 말풍선 안에 여러 줄이 있으면 줄바꿈을 살려서
   하나의 메시지로 합치세요.

5. 여러 장의 이미지가 주어지면 순서대로 이어지는 하나의 대화로 취급하세요.

6. 글자가 흐릿하거나 가려져서 읽을 수 없는 부분은 지어내지 말고 그 메시지를 건너뛰세요.

읽은 대화를 순서대로 messages 배열에 담아 반환하세요."""


# ==========================================================
# 3. 이미지 검증 + 추출
# ==========================================================

class OcrError(Exception):
    """사용자에게 그대로 보여줄 한국어 메시지를 담은 예외입니다."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def _media_type_of(content_type, raw_bytes):
    """
    업로드된 content_type 을 신뢰하지 않고 매직 바이트로 다시 확인합니다.
    브라우저가 잘못된 MIME 을 보내거나, 확장자만 바꾼 파일을 걸러냅니다.
    """

    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"

    if raw_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"

    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "image/webp"

    return None


async def validate_and_encode(files):
    """
    업로드된 이미지 파일들을 검증하고 base64 인코딩까지 마쳐서 돌려줍니다.

    반환: [{"media_type": "image/png", "data": "<base64>"}, ...]
    """

    if not OCR_ENABLED:
        raise OcrError("사진 판독은 지금 사용할 수 없어요.")

    if not files:
        raise OcrError("사진을 선택해 주세요.")

    if len(files) > MAX_IMAGES:
        raise OcrError(f"사진은 한 번에 {MAX_IMAGES}장까지 올릴 수 있어요.")

    encoded = []

    for file in files:

        raw_bytes = await file.read()

        if not raw_bytes:
            raise OcrError("빈 파일이 있어요.")

        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise OcrError("사진 용량이 너무 큽니다. (장당 5MB)")

        media_type = _media_type_of(file.content_type, raw_bytes)

        if media_type is None or media_type not in ALLOWED_MIME_TYPES:
            raise OcrError("PNG, JPG, WEBP 이미지만 올릴 수 있어요.")

        encoded.append(
            {
                "media_type": media_type,
                "data": base64.standard_b64encode(raw_bytes).decode("ascii"),
            }
        )

    return encoded


def extract_messages(images):
    """
    검증·인코딩된 이미지 목록을 받아 [{"speaker","message"}] 를 돌려줍니다.

    images: [{"media_type": "image/png", "data": "<base64>"}, ...]
    """

    if not OCR_ENABLED:
        raise OcrError("사진 판독은 지금 사용할 수 없어요.")

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image["media_type"],
                "data": image["data"],
            },
        }
        for image in images
    ]

    content.append({"type": "text", "text": EXTRACTION_PROMPT})

    try:
        response = _client.messages.create(
            model=OCR_MODEL,
            max_tokens=8000,
            output_config={
                "effort": "medium",
                "format": {
                    "type": "json_schema",
                    "schema": MESSAGES_SCHEMA,
                },
            },
            messages=[{"role": "user", "content": content}],
        )

    except anthropic.APIConnectionError:
        raise OcrError(
            "사진을 읽지 못했습니다. 서버에 연결하지 못했어요. TXT 업로드를 이용해 주세요."
        )

    except anthropic.APIStatusError:
        raise OcrError(
            "사진을 읽지 못했습니다. TXT 업로드를 이용해 주세요."
        )

    if response.stop_reason == "refusal":
        raise OcrError(
            "사진을 읽지 못했습니다. 다른 사진으로 시도하거나 TXT 업로드를 이용해 주세요."
        )

    text_block = next(
        (block for block in response.content if block.type == "text"),
        None,
    )

    if text_block is None:
        raise OcrError("사진을 읽지 못했습니다. TXT 업로드를 이용해 주세요.")

    try:
        parsed = json.loads(text_block.text)
    except ValueError:
        raise OcrError("사진을 읽지 못했습니다. TXT 업로드를 이용해 주세요.")

    messages = [
        {
            "speaker": (item.get("speaker") or "").strip(),
            "message": item.get("message") or "",
        }
        for item in parsed.get("messages", [])
        if (item.get("speaker") or "").strip() and (item.get("message") or "").strip()
    ]

    if not messages:
        raise OcrError(
            "대화를 찾지 못했습니다. 말풍선이 잘 보이는 화면인지 확인해 주세요."
        )

    return messages
