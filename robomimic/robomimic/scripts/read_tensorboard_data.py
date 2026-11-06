from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = "/home/sydney1/Repos/robomimic/trained_models/temporaldp/bc_rnn/can/ph/image/trained_models/temporaldp_bc_rnn_can_ph_image/20260501172620/logs/tb/events.out.tfevents.1777620383.sydney1-MS-7C60"

def get_training_summary(path: str):
    ea = EventAccumulator(path)
    ea.Reload()
    # print("Available tags:")
    # print(ea.Tags())
    
    train_loss = ea.Scalars("Train/Loss")
    steps = max(e.step for e in train_loss)
    start_train_time = min(e.wall_time for e in train_loss)
    last_train_time = max(e.wall_time for e in train_loss)


    try:
        horizon = ea.Scalars("Rollout/Horizon/image_v15")
        success_rates = ea.Scalars("Rollout/Success_Rate/image_v15")

        min_horizon_values = min([e.value for e in horizon])
        max_horizon_values = max([e.value for e in horizon])
        success_rate_values = max([e.value for e in success_rates])
        last_rollout_time = max(e.wall_time for e in horizon)

    except KeyError:
        # potentially no rollout data if training didn't run for very long, or if rollout logging was disabled
        min_horizon_values = None
        max_horizon_values = None
        success_rate_values = None
        last_rollout_time = None

    try:
        validation_loss = ea.Scalars("Valid/Loss")
        last_validation_time = max(e.wall_time for e in validation_loss)
    except KeyError:
        last_validation_time = None

    try:
        last_log_modified_time = (Path(path).parent.parent / "log.txt").stat().st_mtime
    except OSError:
        last_log_modified_time = None


    last_updated_time = [last_train_time, last_rollout_time, last_validation_time, last_log_modified_time]
    last_updated_time = [t for t in last_updated_time if t is not None]
    last_updated_time = max(last_updated_time) if last_updated_time else None
    

    return {
        "max_steps": steps,
        "max_success_rate": success_rate_values,
        "min_rollout_horizon": min_horizon_values,
        "max_rollout_horizon": max_horizon_values,
        "start_time": start_train_time,
        "last_updated_time": last_updated_time
    }

    # for tag in ea.Tags().get("scalars", []):
    #     events = ea.Scalars(tag)
    #     print(f"\n{tag}")
    #     for e in events:
    #         print(f"step={e.step}, value={e.value}, wall_time={e.wall_time}")


if __name__ == "__main__":
    summary = get_training_summary(path)
    print(summary)