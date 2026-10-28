.. _motion-imitation:

Motion Imitation
================

mjlab can train humanoid policies to imitate reference motions. This page
covers motion data preprocessing and training.

WandB registry setup
--------------------

mjlab uses `Weights & Biases <https://wandb.ai/>`_ to store and load
reference motions. Before preprocessing any motions, create a WandB registry
by following the
`BeyondMimic instructions <https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup>`_
(only the registry creation step; skip the ``csv_to_npz.py`` command shown
there).

Motion preprocessing
--------------------

Reference motions are retargeted CSV files in Unitree's generalized
coordinate convention (base position, base quaternion in xyzw, then joint
angles).

Convert a CSV to the NPZ format mjlab expects:

.. code-block:: bash

   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
       --input-file <PATH_TO_CSV> \
       --output-name <MOTION_NAME> \
       --input-fps 30 \
       --output-fps 50 \
       --render True

The script plays the motion through MuJoCo Warp, computes forward kinematics
for every body, and uploads the resulting NPZ to your WandB registry.

``--robot`` selects the robot the CSV was retargeted to (``g1`` by default,
or ``r1`` for the Unitree R1). To keep the NPZ on disk instead of uploading
it, pass ``--output-file`` and ``--upload False``:

.. code-block:: bash

   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
       --robot r1 \
       --input-file <PATH_TO_CSV> \
       --output-name <MOTION_NAME> \
       --output-file motions/<MOTION_NAME>.npz \
       --upload False \
       --input-fps 50

.. warning::

   You **must** use mjlab's converter (``mjlab.scripts.csv_to_npz``).
   Converters from other frameworks such as IsaacLab produce NPZ files with
   incompatible body orderings. The NPZ stores precomputed body positions and
   quaternions indexed by body number, and different physics engines assign
   body indices differently (MuJoCo uses depth first traversal, PhysX uses
   breadth first). A mismatched NPZ will map tracking targets to the wrong
   bodies and training will not converge.

Training
--------

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --registry-name your-org/motions/motion-name \
       --env.scene.num-envs 4096

A local NPZ is passed with ``--env.commands.motion.motion-file`` instead of
``--registry-name``:

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-R1 \
       --env.commands.motion.motion-file motions/<MOTION_NAME>.npz \
       --env.scene.num-envs 4096

Unitree R1 from Axis Studio captures
------------------------------------

R1 reference motions can be produced from Noitom Axis Studio BVH exports
with the `KWoolf/GMR <https://github.com/KWoolf/GMR>`_ fork of GMR, which
retargets to the 26-DoF R1, limits joint velocities, corrects foot
penetration and writes CSVs in the R1 joint order:

.. code-block:: bash

   python scripts/axis_bvh_to_robot.py --bvh_file take.bvh \
       --robot unitree_r1 --save_path take.pkl
   python scripts/robot_motion_to_csv.py --motion_file take.pkl \
       --csv_path take.csv --segment 10 25

``--segment 10 25`` writes the 10 to 25 s window of the take to
``take_10-25s.csv``. The CSV is written at the pickle's frame rate (50 Hz by
default), so convert it with ``--input-fps 50``.

Evaluation
----------

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id
