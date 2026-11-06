"""TemporalDiffusionPolicy benchmark environments.

The environment implementations live in :mod:`waitatgoal` and :mod:`liftqa`.
They are intentionally not imported here: importing LiftQA requires MuJoCo and
robosuite, whereas many robomimic workflows do not need either dependency.
"""
