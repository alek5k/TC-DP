from diffusion_policy_contrib.tools.dataset_video_audio_visualise import export_episode_video_audio

if __name__ == "__main__":
    ds_name = "Piano_b08d56_ckpt20260803_170624_+ID_+SiP128"
    export_episode_video_audio(
        source=f"data/inference/{ds_name}.zarr",
        output=f"./{ds_name}.mp4",
        episode=0,
    )