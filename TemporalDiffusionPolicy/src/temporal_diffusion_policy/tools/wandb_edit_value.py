import os
import wandb

def get_wandb_run_from_name(name: str, wandb_entity: str, wandb_project: str):
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{wandb_entity}/{wandb_project}", per_page=100000)
    for run in runs:
        if run.name == name:
            run_id = run.id
            break
    run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
    return run


def update_run_config(run, key, value):
    run.config[key] = value
    run.update()

def update_run_summary(run, key, value):
    run.summary[key] = value
    run.summary.update()

if __name__ == "__main__":

    wandb_entity = os.environ.get("WANDB_ENTITY")
    if not wandb_entity:
        raise ValueError("Set WANDB_ENTITY to the W&B account or team to update.")
    wandb_project: str = "TemporalDiffusionPolicy"
    
    for run in [
        "chocolate-frog-191",
        "lilac-meadow-201",
        
    ]:
        run = get_wandb_run_from_name(run, wandb_entity, wandb_project)
        update_run_config(run, "img_dim", 224)
        # update_run_config(run, "save_ema_weights", False)
