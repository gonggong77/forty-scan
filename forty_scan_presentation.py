# ==========================================================
# forty-scan
# 영포티 말투 판독기 - 발표용 시각화 스크립트
#
# 목적:
# 로지스틱 회귀 모델이 문장 하나를 입력받아
# 0(정상) / 1(영포티)을 "왜" 그렇게 판정하는지
# 절편 -> 특징별 기여도 누적 -> 시그모이드 확률 로 이어지는
# 계산 과정을 계단식(step) 애니메이션으로 눈에 보이게 만듭니다.
#
# 기존 forty_scan_features.py / forty_scan_model.py / forty_scan_test.py는
# 전혀 수정하지 않고, 이미 저장된 모델(model/forty_scan_hk_model.pkl)과
# 특징추출 함수만 그대로 불러와서 씁니다.
#
# 실행:
#   python forty_scan_presentation.py
#
# 화면 조작:
#   자동으로 재생됩니다.
#   스페이스바 / 클릭 : 일시정지 / 재생
#   -> / <-           : 한 단계씩 수동으로 이동 (일시정지 중에도 동작)
#   q / esc           : 종료
# ==========================================================


# ==========================================================
# 0. 라이브러리
# ==========================================================

import os
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt

from forty_scan_features import FEATURE_NAMES, MODEL_PATH, text_to_multihot


# 윈도우 콘솔에서 한글이 깨지지 않도록 출력 인코딩 지정
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 그래프 안의 한글이 깨지지 않도록 폰트 지정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ==========================================================
# 1. 발표용 예시 문장
#
# 발표에서 보여줄 문장은 여기 두 줄만 바꾸면 됩니다.
# ==========================================================

SENTENCES = [
    ("정상 예시", "가라아게 준다고 해놓고 새우튀김 준거 에바임 진짜 ㅜ ㅜ"),
    ("영포티 예시", "우리 이쁜 ㅇㅇ이~^^ 오빠랑 시원한 초계국수 한 사바리 쐬주 한 잔 콜? ㅎㅎ"),
]


# ==========================================================
# 2. 색상 (dataviz 다이버징 팔레트: blue <-> red, 중립 gray)
#
# 계수/기여도가 양수(영포티 방향)면 red, 음수(정상 방향)면 blue.
# 절편(기본값)은 "검출된 특징"이 아니므로 중립 gray로 구분합니다.
# ==========================================================

COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"

COLOR_NORMAL = "#2a78d6"   # 0(정상) 방향
COLOR_FORTY = "#e34948"    # 1(영포티) 방향
COLOR_NEUTRAL = "#9a988f"  # 절편(기본값) 바 색


# ==========================================================
# 3. 모델 불러오기 (forty_scan_test.py 와 동일한 방식)
# ==========================================================

print()
print("=" * 70)
print("                        FORTY-SCAN")
print("            영포티 말투 판독기 - 발표용 시각화")
print("=" * 70)
print()
print("모델 불러오는 중...")
print(f"      파일 경로 : {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "\n모델 파일을 찾을 수 없습니다.\n"
        f"현재 경로:\n{MODEL_PATH}\n\n"
        "forty_scan_model.py 를 먼저 실행해서 모델을 생성하세요."
    )

saved = joblib.load(MODEL_PATH)
MODEL = saved["model"]
COEF = MODEL.coef_[0]
INTERCEPT = float(MODEL.intercept_[0])

print("      불러오기 완료")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ==========================================================
# 4. 문장 -> 절편 + 특징별 기여도 계산
# ==========================================================


def build_case(label_hint, text):
    """
    문장 하나를 multi-hot 벡터로 바꾸고,
    검출된 특징들을 기여도(등장횟수 x 가중치) 절댓값이 큰 순서로 정리합니다.
    (forty_scan_test.py 의 predict_forty() 와 동일한 계산)
    """

    vector, fuzzy_flags = text_to_multihot(text, return_fuzzy_flags=True)

    contribs = []
    for name, count, coef in zip(FEATURE_NAMES, vector, COEF):
        if count > 0:
            contribs.append(
                {
                    "name": name,
                    "count": int(count),
                    "coef": float(coef),
                    "value": float(count * coef),
                    "is_fuzzy": name in fuzzy_flags,
                }
            )

    contribs.sort(key=lambda item: abs(item["value"]), reverse=True)

    z_final = INTERCEPT + sum(item["value"] for item in contribs)
    prob_final = float(sigmoid(z_final))

    return {
        "label_hint": label_hint,
        "text": text,
        "contribs": contribs,
        "z_final": z_final,
        "prob_final": prob_final,
    }


CASES = [build_case(hint, text) for hint, text in SENTENCES]

for case in CASES:
    label = 1 if case["prob_final"] >= 0.5 else 0
    print()
    print(f"      [{case['label_hint']}] {case['text']}")
    print(
        f"      z = {case['z_final']:+.2f}  ->  "
        f"P(영포티) = {case['prob_final'] * 100:.1f}%  ->  판정 {label}"
    )


# ==========================================================
# 5. 프레임(스텝) 목록 만들기
#
# 프레임 하나 = "이 시점까지 특징을 몇 개 반영했는지"의 스냅샷입니다.
# 부드럽게 보간하지 않고, 값이 하나씩 딱딱 늘어나는 계단식 애니메이션으로
# 판정 과정이 눈에 보이기만 하면 충분합니다.
# ==========================================================

HOLD_FRAMES = 4     # 최종 판정 화면을 몇 프레임 더 붙잡아 둘지
DIVIDER_FRAMES = 3  # 문장 사이 전환 화면 프레임 수
INTERVAL_MS = 1300  # 자동 재생 시 프레임 간 간격 (밀리초)


def build_case_frames(case, case_index):
    frames = []
    contribs = case["contribs"]

    for revealed in range(len(contribs) + 1):
        z_partial = INTERCEPT + sum(c["value"] for c in contribs[:revealed])
        frames.append(
            {
                "type": "step",
                "case_index": case_index,
                "case": case,
                "revealed": revealed,
                "z_partial": z_partial,
                "prob_partial": float(sigmoid(z_partial)),
                "is_final": revealed == len(contribs),
            }
        )

    for _ in range(HOLD_FRAMES):
        frames.append(frames[-1])

    return frames


ALL_FRAMES = []
for i, case in enumerate(CASES):
    ALL_FRAMES.extend(build_case_frames(case, i))
    if i != len(CASES) - 1:
        for _ in range(DIVIDER_FRAMES):
            ALL_FRAMES.append({"type": "divider", "next_case": CASES[i + 1]})

# 시그모이드 x축 범위를 두 문장 모두에서 고정해, 문장이 바뀌어도
# 같은 척도로 비교할 수 있게 합니다.
_all_z = [
    f["z_partial"] for f in ALL_FRAMES if f["type"] == "step"
]
Z_MIN = min(_all_z) - 1.5
Z_MAX = max(_all_z) + 1.5


# ==========================================================
# 6. 그리기
# ==========================================================


def style_axes(ax):
    ax.set_facecolor(COLOR_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_INK_SECONDARY, labelsize=9)


def draw_divider(fig, axes, frame):
    ax_header, ax_bar, ax_sig = axes["header"], axes["bar"], axes["sig"]

    for ax in (ax_header, ax_bar, ax_sig):
        ax.clear()
        ax.axis("off")

    nxt = frame["next_case"]

    ax_bar.text(
        0.5, 0.62, "다음 예시 ▶", transform=ax_bar.transAxes,
        ha="center", va="center", fontsize=24, fontweight="bold",
        color=COLOR_INK_SECONDARY,
    )
    ax_bar.text(
        0.5, 0.38, f"[{nxt['label_hint']}]",
        transform=ax_bar.transAxes,
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=COLOR_MUTED,
    )
    ax_bar.text(
        0.5, 0.22, f"\"{nxt['text']}\"",
        transform=ax_bar.transAxes,
        ha="center", va="center", fontsize=13, color=COLOR_INK, wrap=True,
    )


def draw_header(ax, case, revealed, total, z_partial, prob_partial, is_final):
    ax.clear()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.0, 0.82, f"[{case['label_hint']}]", fontsize=12, fontweight="bold",
        color=COLOR_MUTED, transform=ax.transAxes,
    )
    ax.text(
        0.0, 0.52, f"\"{case['text']}\"", fontsize=15, color=COLOR_INK,
        transform=ax.transAxes,
    )

    if not is_final:
        status = f"특징 반영 중 ({revealed}/{total})    z = {z_partial:+.2f}"
        ax.text(
            0.0, 0.10, status, fontsize=11, color=COLOR_INK_SECONDARY,
            transform=ax.transAxes,
        )
    else:
        label = 1 if prob_partial >= 0.5 else 0
        label_text = "영포티 (1)" if label == 1 else "정상 (0)"
        color = COLOR_FORTY if label == 1 else COLOR_NORMAL
        ax.text(
            0.0, 0.06,
            f"최종 판정 : {label_text}   -   P(영포티) = {prob_partial * 100:.1f}%",
            fontsize=13, fontweight="bold", color="white",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="none"),
        )


def draw_bars(ax, case, revealed):
    contribs = case["contribs"]

    rows = [("절편 (기본값)", INTERCEPT, COLOR_NEUTRAL, False, 0)]
    for c in contribs[:revealed]:
        color = COLOR_FORTY if c["value"] > 0 else COLOR_NORMAL
        rows.append((c["name"], c["value"], color, c["is_fuzzy"], c["count"]))

    ax.clear()
    style_axes(ax)

    y_pos = np.arange(len(rows))
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    ax.barh(y_pos, values, color=colors, height=0.6, zorder=3)

    labels = []
    for name, value, color, is_fuzzy, count in rows:
        tag = f" (x{count})" if count else ""
        note = " *오타보정" if is_fuzzy else ""
        labels.append(f"{name}{tag}{note}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # 위에서부터 절편 -> 기여도 큰 순서

    for y, value in zip(y_pos, values):
        offset = 0.05 * max(1.0, abs(value))
        x_text = value + offset if value >= 0 else value - offset
        ha = "left" if value >= 0 else "right"
        ax.text(
            x_text, y, f"{value:+.2f}", va="center", ha=ha,
            fontsize=9, color=COLOR_INK_SECONDARY,
        )

    all_vals = [INTERCEPT] + [c["value"] for c in contribs]
    span = max(1.0, max(all_vals) - min(0.0, min(all_vals)))
    x_min = min(0.0, min(all_vals)) - span * 0.25
    x_max = max(0.0, max(all_vals)) + span * 0.25
    ax.set_xlim(x_min, x_max)

    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("기여도 (가중치 x 등장횟수)", fontsize=9, color=COLOR_INK_SECONDARY)
    ax.set_title(
        "절편 + 검출된 특징이 하나씩 더해지는 과정", fontsize=10,
        color=COLOR_INK_SECONDARY, loc="left",
    )


def draw_sigmoid(ax, z_partial, prob_partial, is_final):
    ax.clear()
    style_axes(ax)

    x = np.linspace(Z_MIN, Z_MAX, 400)
    y = sigmoid(x)

    ax.axhspan(0.5, 1.02, color=COLOR_FORTY, alpha=0.06, zorder=0)
    ax.axhspan(-0.02, 0.5, color=COLOR_NORMAL, alpha=0.06, zorder=0)
    ax.axhline(0.5, color=COLOR_AXIS, linestyle="--", linewidth=1, zorder=1)

    ax.plot(x, y, color=COLOR_INK_SECONDARY, linewidth=2, zorder=2)

    dot_color = COLOR_FORTY if prob_partial >= 0.5 else COLOR_NORMAL
    dot_size = 160 if is_final else 90

    ax.plot(
        [z_partial, z_partial], [-0.02, prob_partial],
        linestyle=":", color=dot_color, linewidth=1.2, zorder=3,
    )
    ax.plot(
        [Z_MIN, z_partial], [prob_partial, prob_partial],
        linestyle=":", color=dot_color, linewidth=1.2, zorder=3,
    )
    ax.scatter(
        [z_partial], [prob_partial], s=dot_size, color=dot_color,
        edgecolor=COLOR_INK, linewidth=1.2, zorder=4,
    )

    label_y = min(0.97, prob_partial + 0.08)
    span = Z_MAX - Z_MIN
    if z_partial > Z_MIN + 0.8 * span:
        label_ha = "right"
    elif z_partial < Z_MIN + 0.2 * span:
        label_ha = "left"
    else:
        label_ha = "center"
    ax.text(
        z_partial, label_y, f"P(영포티) = {prob_partial * 100:.1f}%",
        ha=label_ha, fontsize=10, fontweight="bold", color=dot_color,
    )

    # 시그모이드는 좌하단 -> 우상단 대각선을 그리므로, 점이 지나가지 않는
    # 반대쪽 구석(좌상단/우하단)에 구간 라벨을 둬서 겹침을 피합니다.
    ax.text(
        Z_MIN + 0.2, 0.94, "영포티(1) 구간", fontsize=9, color=COLOR_FORTY, ha="left",
    )
    ax.text(
        Z_MAX - 0.2, 0.03, "정상(0) 구간", fontsize=9, color=COLOR_NORMAL, ha="right",
    )

    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("z = 절편 + 기여도 합", fontsize=9, color=COLOR_INK_SECONDARY)
    ax.set_ylabel("시그모이드 확률 P(영포티=1)", fontsize=9, color=COLOR_INK_SECONDARY)
    ax.set_title(
        "시그모이드 곡선 위에서 z가 이동하며 확률이 결정되는 과정",
        fontsize=10, color=COLOR_INK_SECONDARY, loc="left",
    )


def draw_frame(fig, axes, frame):
    if frame["type"] == "divider":
        draw_divider(fig, axes, frame)
        return

    case = frame["case"]
    draw_header(
        axes["header"], case, frame["revealed"], len(case["contribs"]),
        frame["z_partial"], frame["prob_partial"], frame["is_final"],
    )
    draw_bars(axes["bar"], case, frame["revealed"])
    draw_sigmoid(axes["sig"], frame["z_partial"], frame["prob_partial"], frame["is_final"])


# ==========================================================
# 7. Figure 구성
# ==========================================================


def build_figure():
    fig = plt.figure(figsize=(10.5, 8.5))
    fig.patch.set_facecolor(COLOR_SURFACE)

    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.6, 2.6], hspace=0.55)

    ax_header = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    ax_sig = fig.add_subplot(gs[2])

    fig.suptitle(
        "FORTY-SCAN  로지스틱 회귀 판정 과정  (0=정상 / 1=영포티)",
        fontsize=15, fontweight="bold", color=COLOR_INK,
    )

    return fig, {"header": ax_header, "bar": ax_bar, "sig": ax_sig}


# ==========================================================
# 8. 실시간 재생 (자동 타이머 + 수동 조작)
# ==========================================================


class LivePresentation:
    def __init__(self, fig, axes, frames, interval_ms):
        self.fig = fig
        self.axes = axes
        self.frames = frames
        self.idx = 0
        self.paused = False

        self.timer = fig.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self.tick)
        self.timer.start()

        fig.canvas.mpl_connect("key_press_event", self.on_key)
        # 스페이스바가 창 포커스 문제로 안 먹는 경우를 대비해,
        # 그래프 아무 곳이나 클릭해도 일시정지/재생이 토글되게 합니다.
        fig.canvas.mpl_connect("button_press_event", self.on_click)

        fig.text(
            0.5, 0.005,
            "스페이스/클릭: 일시정지·재생   ←/→: 한 단계 이동   q: 종료",
            ha="center", fontsize=9, color=COLOR_MUTED,
        )

        self.render()
        self._force_focus()

    def _force_focus(self):
        # VSCode에서 Ctrl+F5로 실행하면, 새로 뜨는 그래프 창이 VSCode 뒤에
        # 가려진 채로 열려서 키보드 입력이 VSCode 쪽으로 가버리는 경우가
        # 많습니다. 창을 맨 앞으로 끌어올리고 포커스까지 강제로 줍니다.
        try:
            window = self.fig.canvas.manager.window
            widget = self.fig.canvas.get_tk_widget()

            def raise_and_focus():
                window.deiconify()
                window.lift()
                window.attributes("-topmost", True)
                window.after(100, lambda: window.attributes("-topmost", False))
                window.focus_force()
                widget.focus_force()

            window.after(200, raise_and_focus)
        except Exception:
            pass

    def render(self):
        draw_frame(self.fig, self.axes, self.frames[self.idx])
        self.fig.canvas.draw_idle()

    def advance(self, step):
        self.idx = (self.idx + step) % len(self.frames)
        self.render()

    def tick(self):
        if not self.paused:
            self.advance(1)

    def on_click(self, event):
        self.paused = not self.paused

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "right":
            self.advance(1)
        elif event.key == "left":
            self.advance(-1)
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


def run_live():
    fig, axes = build_figure()
    LivePresentation(fig, axes, ALL_FRAMES, INTERVAL_MS)
    plt.show()


# ==========================================================
# 9. 진입점
# ==========================================================

if __name__ == "__main__":
    run_live()
