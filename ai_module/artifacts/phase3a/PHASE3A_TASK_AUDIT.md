# Phase 3A Task and VLA Audit

This is an offline audit. No VLA annotation, scene name, object ID, coordinate, or answer is imported by production runtime.

## Challenge grammar

- Public questions compiled: 75/75
- Numerical questions compiled: 15/15 (PASS)
- Scene count: 15

## VLA archive authority

- Path: `/home/robot/cmu_vln/VLA-3D_dataset/Unity.zip`
- Scenes: 15/15
- object_result: 15/15
- region_result: 15/15
- scene_graph: 15/15
- referential_statements: 15/15
- Access: direct `zipfile` reads; no extraction and no archive modification.

## Structural findings

- Region count distribution: `{"count": 15, "max": 9.0, "mean": 2.533333333333333, "median": 2.0, "min": 1.0, "p10": 1.0, "p90": 5.999999999999998}`
- Object count distribution: `{"count": 15, "max": 432.0, "mean": 131.13333333333333, "median": 106.0, "min": 63.0, "p10": 77.80000000000001, "p90": 200.6}`
- Referential records: 119831
- Same-class multiplicity: `{"count": 366, "max": 45.0, "mean": 4.040983606557377, "median": 3.0, "min": 2.0, "p10": 2.0, "p90": 7.0}`
- Candidate domains are formed by class/attribute filters plus finite relation chains; ordered selectors require place-local coverage.
- Color and size are disambiguation evidence, never identity authority.
- Adjacent same-class annotation components must remain distinct unless direct identity evidence supports every merge pair.
