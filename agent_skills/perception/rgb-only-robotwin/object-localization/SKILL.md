---
name: object-localization
description: Ground RoboTwin positions from RGB views and measured robot proprioception.
---

Inspect head and wrist RGB images together with measured EEF poses. Use small,
observable movements to resolve ambiguous alignment. Pixel coordinates alone
do not establish metric object geometry. Mark unknown geometric bounds as
unknown and avoid deriving depth, contact or grasp success from a guessed pose.
Use [robot interface](../robot-interface/SKILL.md) for target conventions.
