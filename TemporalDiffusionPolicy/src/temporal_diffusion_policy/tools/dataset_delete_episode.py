from diffusion_policy_contrib.tools.dataset_delete_episode import delete_episodes

if __name__ == "__main__":
    delete_episodes(
    src="data/demonstration/wait_at_goal.zarr",
    dst="data/demonstration/wait_at_goal_fixed.zarr",
    episodes=[94, 196])