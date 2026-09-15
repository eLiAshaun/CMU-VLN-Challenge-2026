# Phase 3A Recorded N1 Replay

- Source: `/home/robot/cmu_vln/challenge_eval_runs/.ai_runtime_sessions/20260822T223324Z_hotel_room_1_q1_w9vhouv8`
- Completed acquisitions: 11
- First bad acquisition: `20260822T223845_284666Z` (A6)
- Responsible owner: `SAME_ACQUISITION_DUPLICATION`
- Count transition: 4 -> 3
- Escaped panorama containment: 0.999457
- Escaped panorama IoU: 0.911369

The first wrong transition is acquisition-local. SceneMemory receives two objects that should already be one acquisition object; its later anchor-domain pollution is downstream.
