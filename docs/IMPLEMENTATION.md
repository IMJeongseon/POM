# 구현 노트 (v0.2 — depth 기반으로 피벗)

목표: **tri-conditional editing 성공** (source 구조 + appearance 외형 + text 배경).
PDG의 residual 분해/완전 복원 요구를 버리고, 선행연구(A Training-Free Framework for
High-Fidelity Appearance Transfer) 스타일로 전환.

## 아키텍처 (피벗 후)

```
구조(structure)  = source depth (SD3 ControlNet-Depth, 매 스텝) + Blended Noise Init(replay_k)
외형(appearance) = reference K/V를 vital layer에 Attention Context Expansion,
                   velocity-level mask routing으로 전경 국한
배경(background) = target text prompt + CFG(prompt 준수)
```
per-step (2-pass, mask routing):
```
v_bg = CFG( v(x,s,c_target | depth) )                  # 프롬프트+구조
v_fg = CFG( v(x,s,c_target | depth, inject ref K/V) )  # +외형
v    = M*v_fg + (1-M)*v_bg
```

## 모듈

| 파일 | 역할 |
|---|---|
| `pipeline.py` | tri-conditional edit loop (2-pass mask routing) |
| `anchor.py` | inversion(midpoint/fireflow/euler) + Blended Noise Init 궤적 cache |
| `depth.py` | Depth-Anything-V2 → depth 조건 이미지 |
| `masks.py` | BiRefNet 전경 마스크 |
| `routing.py` | velocity-level region routing + latent mask feather |
| `drift.py` | drift-gated 주입 (`d_i`) |
| `solver.py` | Euler step |
| `backends/sd3_depth.py` | **SD3 + ControlNet-Depth + CFG** (주 백본) |
| `backends/sd3.py`, `flux.py` | 플레인 SD3 / FLUX |
| `backends/_mmdit_attn.py` | to_k/to_v hook 기반 K/V capture + ACE injection |
| `backends/dummy.py` | 가중치 없는 테스트용 |

제거됨: `residual.py`, `schedules.py` (velocity-residual 분해 — 피벗으로 폐기).

## 검증 상태 (conda `pdg`: torch 2.8+cu128, diffusers 0.35.1, RTX PRO 6000 96GB)

- **코어**: `pytest tests/test_core.py` → **7/7 pass** (routing, drift, inversion round-trip, anchor, full loop).
- **부품 GPU 검증**: Depth-Anything-V2 ✅, BiRefNet(전경 0.22) ✅, SD3-Controlnet-Depth forward ✅
  (depth가 velocity를 바꿈 diff 0.10), SD3 K/V capture+ACE ✅.
- **End-to-end tri-conditional**: SD3+ControlNet-Depth로 **실행 성공** (768px, ~11s/이미지).
  - ✅ **구조 보존**: depth로 기린 형태 정확히 유지.
  - ⚠️ **배경(text)**: CFG로 부분 작동 (겨울 기운은 들어오나 약함; CFG scale·replay_k 튜닝 필요).
  - ⚠️ **외형(appearance)**: 주입이 출력에 **확실히 영향**을 주나 강도가 미묘 —
    mid layer는 너무 약하고(`strength≈1` 미시각), first/last layer나 `strength≳1.5`는
    전경에 blocky 아티팩트. **clean sweet spot은 추측 불가 → vital-layer 스캔 필요.**
  - 결과 이미지: `outputs/tri_cfg.png`(약), `outputs/tri_mask.png`(BiRefNet), `outputs/tri_strong.png`/`tri_s1.5.png`(과주입 아티팩트).

## Appearance 전이 — 실험 로그 & 핵심 발견

- raw K/V concat (mid layer): 약함(미시각) / (first·last layer): blocky 아티팩트.
- **Purified K/V** (reference 전경 마스크, §2.2 구현): strength 1.0 깨끗하나 외형 여전히 약함, 1.5 아티팩트.
- **Redux-analog** (CLIP-L 이미지 임베딩 → pooled 톤 블렌드, alpha 0.5/0.8): **너무 약함**.
  - 원인: 진짜 Redux/IP-Adapter는 **학습된 projector**로 이미지 토큰을 cross-attn에 주입. 학습 없이
    정렬되는 신호는 pooled(CLIP-L 768/2048, 전역 약 modulation)뿐 → 약함. **강한 image-prompting은
    본질적으로 학습된 어댑터 필요.** SD3-medium용 IP-Adapter는 캐시에 없음(InstantX IP-Adapter는
    SD3.5-large+SigLIP).

**Vital-layer 스캔 결과 (해결!)** — `scripts/vital_layer_scan.py` (anchor 1회 빌드 후 layer별 edit):
SD3-medium 24블록 중 **layer 5-8이 clean 강전이 vital layer**. 4-5=blocky, 6-7=깨끗한 leopard 반점,
10-11=tiger 줄무늬, mid(8-15)=약함, first/last=아티팩트. (주의: CLIP-L 전역 유사도 지표는 텍스처
전이에 둔감 — 기린·치타가 이미 cos 0.71 — **육안 확인이 필수**였음.)
→ 기본값: `injection.layers=[5,6,7,8]`, `strength≈1.8`, `appearance_alpha≈0.3`, drift gate off.

**⚠️ 통제실험으로 외형 전이 실패 확인 (outputs/ctrl_grid.png)**: source=기린 고정 + 중립 프롬프트로
appearance만 zebra/tiger/leopard/cow 교체 → 출력이 전부 거의 동일한 기린(appearance 안 따라감).
즉 **K/V injection(ACE)+Redux-analog는 충실한 cross-subject 외형 전이를 못 함**. cheetah→기린
"성공"은 착시(치타≈기린 반점 + appearance_prompt 텍스트 누설). layer 5-8은 약한 텍스처 교란일 뿐.
**진짜 해법 = correspondence 기반 주입**(Eye-for-an-eye/FreeGraftor: dense 대응으로 reference feature를
대응 위치에 재배치) 또는 학습된 adapter(Redux/IP-Adapter, FLUX).

## 도메인 일반화 & 배경 (2026-06-01)

`scripts/make_grid.py` → source|appearance|output + 프롬프트 캡션 (outputs/grid_*.png).
- **animals/birds**: 외형(반점/날개)·배경(눈) 전이 보임 ✅. **cars/fish**: 구조 ✅이나 외형(rigid·저텍스처)·배경 약함.
- **배경 약함 원인**: (1) depth를 전 이미지에 걸면 배경 geometry가 source에 고정, (2) Blended Noise Init이 source 배경까지 replay.
- **적용한 수정**: depth를 **전경 마스크로 제한**(`set_structure_image(image, foreground_mask)`, 배경 depth 평탄화) → coherent + 배경 약간 자유로워짐(birds 눈 배경 개선).
- **`free_background`(opt-in, 기본 off)**: 배경을 sigma=1 노이즈에서 재생성. 강한 배경 변화(차 눈장면) 가능하나 전경 충실도/외형과 상충하고 finicky(replay_k와 동반 튜닝 필요). foreground-pin replay도 시도했으나 regression. → **기본 off, 추가 연구 과제**.

## Appearance 튜닝 (다음 핵심 실험)

외형 전이를 깨끗하게 만들려면 (선행연구도 강조):
1. **Stable-Flow vital-layer 스캔** — SD3 24블록 중 외형을 *깨끗이* 옮기는 layer 집합 탐색
   (현재: mid=약함, first/last=아티팩트). `injection.layers` 재설정.
2. **strength·window 스케줄** — `injection.strength`(현 1.0)와 `window`를 layer별로 튜닝.
3. **drift gate 재도입** — 외형 유지 위해 thresholds 완화했으나, 아티팩트 억제용으로 중간값 재설정.
4. **purified appearance feature** (선행연구 Redux mask-weighted) — SD3엔 Redux 없음 → 대안 검토.

## 실행 (conda pdg)

```bash
conda activate pdg
python -m pytest tests/test_core.py -q                       # 코어
python scripts/verify_backend.py --backend sd3 --res 512     # 백본 스모크
python scripts/reconstruct.py --backend sd3 --source data/animals/animal0.png  # 복원 PSNR
# tri-conditional 편집: scripts/run_edit.py (config: configs/sd3_default.yaml)
```

## Source 복원 (anchor inversion) — 실측

inversion= midpoint(RK2). euler ~16dB(불가), midpoint+소스캡션 ~34dB(VAE천장 39.8). 빈 프롬프트는 불안정.
단 피벗 후엔 **완전 복원이 필수 아님** — depth가 구조를 잡으므로 anchor는 Blended Noise Init용 content prior로 충분.
