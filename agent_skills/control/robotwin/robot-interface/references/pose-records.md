# Pose evidence

Store the episode id, recorded_frame, arm, measured world XYZ+WXYZ, gripper
opening, image references and the observation that justified a target. A target
command is not a measured achieved pose. Do not reuse coordinates across scenes
without re-grounding their reference frame. Constraint annotations carry their
own frame, interval and evidence in the episode HDF5.
