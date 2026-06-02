# PDG → Flow Matching 확장 설계 문서

> **상태**: Draft v0.2 (2026-06-01) — §6 아키텍처 Codex 의논 반영(ODE trajectory cache,
> velocity-level region routing, residual norm-clamp, drift-gated 주입, prompt-diff 공통 축)
> **목표**: PDG(diffusion / PixArt-α / DDPM)를 Flow Matching 백본(FLUX·SD3)으로 이식한
> zero-shot, training-free, tri-conditional 이미지 제어 프레임워크 설계.

---

## 1. 최종 목표 (GOAL)

**Source Image + Appearance Image + Text Prompt → tri-conditional 제어.**

| 조건 | 역할 | 예시 |
|---|---|---|
| Source Image | 전경 **구조(geometry)** 보존 | 기린 |
| Appearance Image | 전경에 **외형(appearance)** 전이 | 얼룩말 무늬 |
| Text Prompt | **배경 의미(semantics)** 편집 | "snowy winter" |

결과: source의 구조 위에 appearance가 입혀진 **얼룩무늬 기린**이 **눈 오는 겨울 초원**에 있는 모습.

**제약**
1. **Zero-shot / training-free** — 추가 학습·fine-tuning 없이 사전학습 백본만 사용.
2. **강력한 백본 의존** — FLUX, SD3 (둘 다 병행).
3. **Flow Matching 기반** — diffusion(DDPM) 버전은 PDG에서 완료. 이를 FM으로 확장.

---

## 2. PDG (기준선) 요약

PDG = **Prediction-space Decomposition Guidance** (DDPM / PixArt-α).

1. **DDPM inversion** (edit-friendly, stochastic) → 구조 보존 anchor 궤적.
2. frozen DiT를 4-pass로 probing: **Null / Base / Inject / Cache**.
3. ε-space residual 추출:
   - `r_bg  = ε_base − ε_null`   (배경 텍스트 방향)
   - `r_app = ε_inj  − ε_base`   (전경 외형 방향)
4. 재구성: `ε_final = ε_base + λ_bg·r_bg + γ_app·r_app`.
5. 특정 layer(18–27/28)·step(5–44)에서만 self-attn **K,V 주입** + attention contrasting + AdaIN.
6. BiRefNet 전경 마스크로 region routing.

핵심 결과: 구조 보존 mIoU **0.960**.

---

## 3. 핵심 등식: 무엇이 "그대로" 넘어가는가

SD3/FLUX 컨벤션 `z_t = (1−t)·x_0 + t·ε`, 예측 속도 `v = ε − x_0`.
같은 `(z_t, t)`에서:

```
ε   = z_t + (1−t)·v
x_0 = z_t − t·v
⇒  ε_A − ε_B = (1−t)·(v_A − v_B)
```

**결론**: residual의 **방향(direction)은 ε-space와 v-space에서 동일**하고, 크기만 `(1−t)` 스칼라
차이 (`z_t`가 상쇄됨). 이것이 이식 가능성의 분기점이다.

---

## 4. 이식 가능성 점검 (Transfer Audit)

> 한 줄 결론: **velocity 치환(수식 한 줄)은 옮겨지지만, PDG를 PDG로 만든 나머지
> machinery(inversion 구조 anchor · attention 주입 · 스케줄 · FLUX-CFG)는 그대로 안 된다.**

| 요소 | 이식 가능성 | 비고 |
|---|---|---|
| residual decomposition **방향** | ✅ 보존 | `ε`-res `= (1−t)·v`-res |
| 계수 λ_bg, γ_app | ⚠️ 재튜닝 | `(1−t)` 소실 → 고노이즈 스텝 과대평가; DDPM↔FM 스텝 매핑 비선형 |
| 샘플러 업데이트 | ⚠️ 재설계 | DDPM stochastic(ᾱ, σ_t·z) → deterministic ODE Euler |
| 구조 anchor (DDPM inversion) | ❌ 직접 대응물 없음 | RF는 deterministic 역적분, per-step noise map 부재 → 교체·재검증 |
| K,V 주입 지점 | ❌ 재유도 | PixArt 분리형 self-attn → FLUX/SD3 **joint attention**의 I2I 서브블록 |
| layer/step/타임스텝 의미 | ❌ 재캘리브레이션 | vital layer 재탐색 + few-step 재스케줄; DDPM ᾱ ↔ RF 직선경로 |
| `r_bg` on FLUX | ❌ 우회 설계 | FLUX-dev guidance-distilled → `v_null` 비표준 (SD3는 정상) |
| disentanglement 성립 | ❓ 실험 검증 | v-space 세 조건 separability는 백본 거동 의존 |

### 4.1 ① 방향만 보존 — 계수·업데이트는 재설계
- residual decomposition **개념**(`r_bg`, `r_app` 방향)은 살아남음.
- 상수 계수 `λ_bg, γ_app`를 그대로 쓰면 안 됨: `(1−t)` 인자 소실로 v-space에서 고노이즈(초기)
  스텝 guidance가 상대적으로 과대평가됨 (`t→1`에서 `(1−t)→0`). DDPM↔FM 스텝 매핑도 비선형 → **계수 전면 재튜닝**.
- 샘플러: PDG는 `ε_final`을 DDPM update에, FM은 `v_final`을 ODE Euler(`x_{t−Δ}=x_t−Δ·v`)에
  넣음. 방향이 같아도 "예측 → 다음 latent" 사상이 다름.

### 4.2 ② 재유도 필요 (진짜 gap)
- **Inversion / 구조 anchor (최대 위험)**: PDG의 edit-friendly DDPM inversion은 stochastic이며
  매 스텝 noise map에 구조를 각인하고 높은 분산으로 편집 robustness를 확보. RF inversion은
  deterministic ODE 역적분이라 per-step stochastic noise map 자체가 없음 → 핵심 장점 소실.
  **교체 후보**: FireFlow(정확 ODE inversion) / SNR-Edit(structure-aware noise prior) /
  FlowEdit(inversion-free). 성질이 서로 다른 별개 기법 → mIoU 0.96급 구조 보존 **재검증 필요**.
- **Attention 주입 지점**: PixArt는 이미지 self-attn(구조)과 텍스트 cross-attn 분리. FLUX/SD3는
  joint attention이라 "이미지 전용 self-attn" 부재 → ReFlex처럼 **I2I 서브블록만 외과적으로
  타겟팅**(재유도). FLUX는 double-stream + single-stream 이종 구조 → "layer 18–27" 무의미.
- **Layer/step + 타임스텝 의미**: vital layer를 Stable Flow로 재탐색, few-step(FLUX ~28)로
  재스케줄. DDPM ᾱ 스케줄의 "어느 스텝이 구조/외형/의미를 담는가"가 RF 직선 경로에서 달라짐 → 재캘리브레이션.

### 4.3 ③ FLUX에서 구조적으로 깨짐
- FLUX-dev는 guidance-distilled → `v_null`이 표준 unconditional이 아님 → `r_bg=v_base−v_null`이
  깨끗한 CFG residual이 아님. **SD3는 진짜 CFG라 정상**.
- 두 백본 병행 시 FLUX 경로는 prompt-difference / attention-level residual(ReFlex·FluxSpace식)로 **우회 설계** 필수 → 동일 코드로 안 돌아감.

### 4.4 ④ 미해결 이론 질문
- v-space에서 세 조건 **disentanglement**가 성립하는가? `(1−t)` 재가중으로 외형/배경 residual
  상대 비중이 t에 따라 변하고, RF 직선 경로에서는 구조·외형 출현 타이밍이 DDPM과 다름.
  matched-residual probing 가정(동일 latent 공유 → residual이 한 factor만 분리)은 샘플러
  무관하게 성립하나, **factor separability 자체는 백본 거동에 의존** → 보장 안 됨, 실험 검증 대상.

---

## 5. 컴포넌트별 매핑 & 차용 레퍼런스

| PDG (DDPM / PixArt-α) | FM 확장 (FLUX·SD3) | 차용 building block |
|---|---|---|
| ε-residual decomposition | **v-residual decomposition** | CFG-in-velocity (SD3) |
| DDPM inversion → 구조 anchor | RF ODE inversion **또는** inversion-free | **FireFlow** / **SNR-Edit** / FlowEdit |
| 4-pass probing | deterministic ODE, 동일 상태 평가 | — |
| self-attn K,V 주입 (18–27) | **joint-attn I2I 성분에만 주입** | **ReFlex**, **A Training-Free DiT**(Attention Context Expansion) |
| layer/step 수동 선택 | vital layer 자동 탐색 + few-step | **Stable Flow** |
| BiRefNet 전경 마스크 | 그대로 | FreeCus / FreeGraftor |
| attention contrasting + AdaIN | 그대로(백본 튜닝) | Cross-Image / Eye-for-an-eye |
| 외형 정렬 (포즈 미정렬 대응) | dense correspondence 기반 재배열 (선택) | Eye-for-an-eye / FreeGraftor / DIFT |

---

## 6. 제안 아키텍처 (Codex 의논 반영)

> **핵심 재정의**: DDPM의 "stochastic noise-map replay + 자유로운 per-step injection"을
> 흉내 내지 않는다. FM에서는 **deterministic ODE trajectory를 anchor로 cache**하고,
> 매 integration point에서 vector field 자체를 조건별 probing으로 수정한다(`F̃`).
> PDG의 prediction-space 분해는 **버리는 게 아니라 velocity-field probing 형태로 재해석**한다
> (separability가 tri-conditional contribution의 핵심이므로 유지).

### 6.1 Anchor = ODE trajectory cache (noise map 아님)

```
anchor = { tᵢ, xᵢ^A, vᵢ^A, 선택 layer Kᵢ^A/Vᵢ^A, featureᵢ^A }   for i = n..k
```

- source를 **FireFlow inversion** → 동일 scheduler/grid로 source prompt 하에 **deterministic
  replay**하며 latent·velocity·선택 layer K/V·hidden 저장.
- 재현성은 noise map이 아니라 `same x_T + same grid + same solver + same conditioning`로 보장.
- **2-branch**: 고정 source anchor branch(`x^A`) + edit branch(`x^E`, 매 step `F̃`로 적분).
- **partial replay**(step `k`까지만)로 편집 강도 조절 — `k`가 noise 쪽일수록 편집 자유도↑.

### 6.2 매 step: probing → velocity residual 합성

```
v_base = v(zₜ, t, c_text)                        # base
v_app  = v(zₜ, t, c_text | appearance K,V inj)   # 외형 주입 pass
# 텍스트/배경 방향 — 백본별로 다르게 (§7):
r_txt  = v(zₜ,t,c_tar) − v(zₜ,t,c_src)           # FLUX·SD3 공통 (prompt-difference, FlowEdit식)
#  또는 SD3: r_bg = v_base − v_null               # 진짜 CFG 가능 시
r_app  = v_app − v_base                          # 외형 방향

# 안정화: residual norm clamp (||v_base|| 기준)
r_txt ← clamp_norm(r_txt, α·||v_base||)
r_app ← clamp_norm(r_app, β·||v_base||)

v_ctrl = v_base + λ(t)·r_txt + γ(t)·r_app
```

### 6.3 Region routing은 velocity 단계에서 (latent blending 아님)

```
v_final = M_fg·v_ctrl + (1 − M_fg)·v_anchor       # 배경은 anchor velocity로 보존
z_{t−Δ} = solver_step(zₜ, v_final)                # Euler 우선
```

- `M_fg`: BiRefNet 전경 마스크 → **latent 해상도로 downsample + feathered**(hard mask는 ODE drift↑).
- `λ(t), γ(t)`: **t-의존 step window** — early=구조/layout, mid=identity/shape, late=texture/color.
  `(1−t)` 보정(§3) + few-step 재스케줄 반영.

### 6.4 Drift-gated 주입 (ghosting 방지)

edit latent가 anchor에서 벌어진 정도 `dᵢ = ||xᵢ^E − xᵢ^A|| / ||xᵢ^A||`에 따라 주입 강도 조절:

```
dᵢ 작음  → 강한 K/V injection
dᵢ 중간  → masked K/V 또는 attention-map bias만
dᵢ 큼    → velocity/feature residual 중심, raw K/V injection 축소
```

- 외형 K/V 주입은 Stable Flow vital layer + ReFlex mid-step robust feature의 **특정 layer/window에 한정**.

### 6.5 Solver 선택

- **Euler 또는 FireFlow식 single-eval 우선** — per-step probing·K/V injection과 가장 잘 맞음.
- midpoint/RK2는 `anchor_mid` cache가 필요하고 K/V 보간이 의미 불분명 → **나중에 확장**.
- solver 자체를 바꾸지 말고, model 호출을 `F̃`(probing+합성)로 **감싸는** 방식이 안전.

---

## 7. 백본 병행 전략

| 항목 | SD3 | FLUX |
|---|---|---|
| CFG / 텍스트 residual | 진짜 CFG → `r_bg=v_base−v_null` + prompt-diff | guidance-distilled → **prompt-diff 중심** + guidance-scalar diff + attn/feature residual |
| 권장 위치 | residual 분해 검증의 **기준 구현** | SD3 검증 후 우회 경로로 확장 |
| 블록 구조 | MM-DiT 균일 | double + single stream 이종 |
| 권장 inversion | FireFlow / SNR-Edit | FireFlow (FLUX 대상 검증됨) |

**텍스트/배경 residual 우선순위 (Codex 권장)**
- **SD3**: true CFG residual + prompt-difference + feature injection
- **FLUX-dev**: prompt-difference(`v_target − v_source`, FlowEdit식) + guidance-scalar diff + attention/feature residual.
  `v_null`은 비추천(guidance-distill로 semantic-pure하지 않음). de-distilled/true-CFG 변형은 training-free·백본 일반성 목표와 충돌.
- → **prompt-difference residual을 공통 1차 축으로** 삼으면 두 백본에서 동일 코드 경로 확보.

→ **공통 인터페이스**(velocity predictor, attn-injection hook, mask router, ODE sampler)를
추상화하고 백본별 어댑터로 CFG/블록 차이를 흡수.

---

## 8. 열린 리스크 & 검증 계획

1. **구조 보존 재현 (최우선)**: RF inversion(FireFlow vs SNR-Edit vs FlowEdit)별 mIoU 비교 →
   PDG의 0.96 재현 가능 여부.
2. **Disentanglement**: v-space residual이 구조/외형/배경을 실제로 분리하는지 ablation.
3. **계수 스케줄**: `λ(t), γ(t)`의 t-의존 step window 형태 탐색 (상수 대비) + `(1−t)` 보정 검증.
4. **FLUX 우회 residual**: `v_null` 부재 시 prompt-difference / guidance-scalar / attention-level
   residual의 품질 비교.
5. **vital layer**: SD3 / FLUX 각각에서 외형 주입에 유효한 layer 집합 재탐색.
6. **anchor-edit drift**: `dᵢ = ||x^E−x^A||/||x^A||` 추적 → drift-gated 주입이 ghosting/texture
   tearing을 실제로 억제하는지 검증.
7. **region routing 방식**: velocity-level 라우팅 vs latent blending 비교 (drift·경계 아티팩트).
8. **solver 정합성**: Euler/FireFlow-wrap 안정화 후, midpoint/RK2(+`anchor_mid` cache) 확장 시
   K/V injection과의 충돌 여부.
9. **residual norm clamp**: `α, β` (clamp 비율) 민감도 — 미적용 시 폭주 여부.

---

## 9. 다음 단계

- [ ] **ODE trajectory cache(anchor) 모듈** — FireFlow inversion + partial replay(`k`), per-t
      latent/velocity/K-V/feature 저장 (§6.1)
- [ ] SD3 기반 velocity-residual 프로토타입 (prompt-diff 공통 축, 기준 구현)
- [ ] joint-attention I2I 주입 hook 설계 (ReFlex 분석) + drift-gated 게이팅(§6.4)
- [ ] velocity-level region router (feathered latent mask)
- [ ] FLUX 우회 residual(prompt-diff / guidance-scalar / feature) 검증
- [ ] 평가 벤치마크 / 메트릭 정비 (mIoU, CLIP-I/T, user study)

---

## 참고 (references/ 자산 매핑)

- **이론 기반**: Flow Matching, Rectified Flow, Stable Diffusion 3, Rectified Flow Prior
- **Inversion / flow editing**: FireFlow, FlowEdit, FlowOpt, ReFlex, SNR-Edit, Stable Flow
- **Appearance / subject transfer**: Cross-Image Attention, Eye-for-an-eye,
  A Training-Free DiT Appearance Transfer, FreeCus, FreeGraftor, ReStyle3D, FluxSpace
- **조건부 제어**: OminiControl, FLUX.1 Kontext, DiffEditor
- **대응**: Emergent Correspondence (DIFT)
- **기준선**: PDG
