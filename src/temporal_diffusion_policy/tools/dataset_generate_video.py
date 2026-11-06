from diffusion_policy_contrib.tools.dataset_generate_video import generate_merged_video



if __name__ == "__main__":
    generate_merged_video(
        path="data/inference/WaitAtGoal_example.zarr",
        image_key="full_image",
        episodes=[1],
        write_episode_index=False
    )