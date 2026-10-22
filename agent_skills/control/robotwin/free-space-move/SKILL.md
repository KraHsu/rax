---
name: free-space-move
description: Plan native RoboTwin EEF target commands and inspect the resulting motion.
---

Use `robotwin_step_eef` with absolute world XYZ+WXYZ targets. RoboTwin's native
planner handles the target; this adapter does not provide collision-aware scene
previews or a separate plan token. Check a fresh observation after each short
move and use visible free space. Refer to [robot interface](../robot-interface/SKILL.md).
