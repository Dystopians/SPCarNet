# L5 cache Pareto — quality vs TOTAL artifact MB (final stack)

| scene | point | PSNR | LPIPS | cache MB raw/comp | ckpt MB | TOTAL raw MB | dPSNR vs uncompressed |
|---|---|---|---|---|---|---|---|
| garden | uncompressed | 26.370 | 0.1286 | 1333/1207 | 942 | 2275 | — |
| garden | jpeg95 | 26.274 | 0.1362 | 851/725 | 942 | 1793 | -0.095 |
| garden | jpeg85 | 26.205 | 0.1467 | 771/645 | 942 | 1713 | -0.165 |
| garden | jpeg70 | 26.045 | 0.1637 | 738/612 | 942 | 1680 | -0.324 |
| garden | halfres | 25.517 | 0.1970 | 331/300 | 942 | 1273 | -0.853 |
| garden | ksubset50 | 25.956 | 0.1412 | 663/600 | 942 | 1605 | -0.414 |
| bicycle | uncompressed | 23.713 | 0.2777 | 1216/1104 | 908 | 2124 | — |
| bicycle | jpeg95 | 24.028 | 0.2615 | 812/700 | 908 | 1720 | +0.315 |
| bicycle | jpeg85 | 23.993 | 0.2692 | 745/632 | 908 | 1653 | +0.280 |
| bicycle | jpeg70 | 23.974 | 0.2778 | 717/605 | 908 | 1625 | +0.261 |
| bicycle | halfres | 23.468 | 0.3277 | 312/284 | 908 | 1221 | -0.245 |
| bicycle | ksubset50 | 23.681 | 0.2877 | 604/546 | 908 | 1512 | -0.032 |
| kitchen | uncompressed | 30.486 | 0.1321 | 2631/2324 | 709 | 3340 | — |
| kitchen | jpeg95 | 30.405 | 0.1360 | 1786/1479 | 709 | 2494 | -0.081 |
| kitchen | jpeg85 | 30.278 | 0.1421 | 1659/1352 | 709 | 2367 | -0.208 |
| kitchen | jpeg70 | 30.139 | 0.1483 | 1611/1304 | 709 | 2320 | -0.347 |
| kitchen | halfres | 29.541 | 0.1740 | 688/616 | 709 | 1397 | -0.945 |
| kitchen | ksubset50 | 29.771 | 0.1367 | 1317/1162 | 709 | 2026 | -0.715 |

TOTAL = checkpoint + raw cache (the on-disk-usable artifact); lossless-compressed cache size also listed (shippable form).
CIs for any headline point: rerun tools/ecr/rung_gate.py on the chosen pair.
