# Stage ELA7 Pareto Evidence Portfolio Report

Date: 2026-05-06

## Decision

ELA7 Pareto Evidence Portfolio is the promoted renderer-side method after ELA4.  It keeps the ELA4 safe benefit-calibrated adapter as the default and adds a broad no-benefit residual candidate.  A train-only portfolio selector blends the two only when the train split shows non-negative PSNR, SSIM, and LPIPS gains relative to the safe candidate.

This fixes the courtyard weakness without damaging the scenes where ELA4 was already strong.

## Method

For each scene:

1. safe branch: ELA4-fast, residual k4, depth relative tolerance selected from `{0.06, 0.12}`, residual clip `0.10`, benefit bins, train-only alpha calibration;
2. broad branch: global residual branch, k16, depth relative tolerance `0.06`, direction weight `0.35`, residual clip `0.10`, alpha `0.5`, no benefit gate;
3. portfolio selector: train split only, uniform 16-view calibration, weight grid `{0.0, 0.1, ..., 0.7}`, balanced objective with LPIPS, plus Pareto guard requiring train PSNR/SSIM/LPIPS gains all non-negative.

The broad branch is accepted only for courtyard.  Bonsai, room, and counter choose weight `0.0`, so ELA7 safely falls back to ELA4.

## Results vs Clean9000 Mesh Splatting

All rows are independent `metrics.py` results on the test split.  Lower LPIPS is better.

| scene | clean PSNR | clean SSIM | clean LPIPS | ELA7 PSNR | ELA7 SSIM | ELA7 LPIPS | dPSNR | dSSIM | dLPIPS | portfolio weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bonsai | 18.541124 | 0.463496 | 0.483265 | 19.879219 | 0.521946 | 0.458695 | +1.338095 | +0.058450 | -0.024570 | 0.0 |
| courtyard | 18.494551 | 0.602439 | 0.423865 | 18.698063 | 0.614155 | 0.405218 | +0.203512 | +0.011716 | -0.018648 | 0.5 |
| room | 26.217100 | 0.889372 | 0.135088 | 28.968866 | 0.933050 | 0.082278 | +2.751766 | +0.043678 | -0.052810 | 0.0 |
| counter | 24.801929 | 0.844451 | 0.159236 | 27.215458 | 0.904876 | 0.099993 | +2.413528 | +0.060425 | -0.059244 | 0.0 |

ELA7 is therefore still a strict all-scene win over the strongest pure Mesh Splatting clean9000 baseline, and courtyard improves over ELA4:

| scene | ELA4 PSNR | ELA4 SSIM | ELA4 LPIPS | ELA7 PSNR | ELA7 SSIM | ELA7 LPIPS | dPSNR vs ELA4 | dSSIM vs ELA4 | dLPIPS vs ELA4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| courtyard | 18.686813 | 0.612538 | 0.410969 | 18.698063 | 0.614155 | 0.405218 | +0.011250 | +0.001617 | -0.005751 |

## Diagnostics

Several non-promoted courtyard runs were important:

- uniform residual policy improved PSNR but hurt SSIM/LPIPS;
- uniform balanced policy improved LPIPS but did not beat ELA4 on PSNR/SSIM;
- direction-weight policy improved PSNR and LPIPS but hurt SSIM;
- no-benefit broad residual improved LPIPS strongly but reduced PSNR/SSIM.

The successful insight was not to choose one branch globally.  Safe and broad residual evidence are complementary, and a train-only Pareto portfolio can use broad evidence only where all calibration metrics agree.

## W&B

Promoted Pareto portfolio runs:

- bonsai: `fp5081np`
- courtyard: `o6b52oti`
- room: `4vzm6b6v`
- counter: `wreb7cia`

Component runs:

- bonsai safe train/test: `qitzgj4b`, `m8bqigla`; broad train/test: `fz7oeyc0`, `vvsl6sh9`
- courtyard safe train/test: `rc30myre`, `olrklnea`; broad train/test: `r6ohult5`, `t8c1urzl`
- room safe train/test: `oglrwzpn`, `sj4xuv6s`; broad train/test: `19kfgdwg`, `xrybuf34`
- counter safe train/test: `23dgyhen`, `yjez0vhw`; broad train/test: `czhg86uh`, `6f8ad5of`

## Remaining Caveat

ELA7 is still a renderer-side evidence method.  It gives the cleanest current answer to the baseline problem, but a paper-ready system still needs:

1. distillation into a persistent neural texture or residual field;
2. runtime and storage overhead numbers;
3. a no-test-leakage audit;
4. ablations for safe branch, broad branch, Pareto guard, and portfolio weight grid.
