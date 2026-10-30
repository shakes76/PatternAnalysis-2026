# ADNI split audit

Total: 680 subjects, 1526 scans, 30520 slices (20 slices per scan).
Label coding in metadata: 0 = NC, 2 = AD.

## Provided split leaks
- Provided train subjects: 565
- Provided test subjects: 331
- Subjects in BOTH: 216 (65% of test subjects)

## Subject-level split used (seed 42, stratified by label)
| Split | Subjects | Scans | Slices | NC scans | AD scans |
|---|---|---|---|---|---|
| train | 475 | 1075 | 21500 | 547 | 528 |
| val | 102 | 231 | 4620 | 114 | 117 |
| test | 103 | 220 | 4400 | 122 | 98 |
