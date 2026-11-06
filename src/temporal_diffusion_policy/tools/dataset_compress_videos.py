from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from diffusion_policy_contrib.tools.dataset_compress_video import compress_video_to_dataset

def _compress_one(dataset_path_str: str, output_path_str: str) -> dict:
    return compress_video_to_dataset(
        input_dataset=dataset_path_str,
        output_dataset=output_path_str,
        image_keys=("full_image", "hand_image"),
        fps=30.0,
        video_lossless=True,
        require_confirmation=False,
    )

if __name__ == "__main__":
    data_dir = Path.cwd() / "data_saved"
    
    inference_datasets = data_dir / "inference"
    inference_compressed_dir = data_dir / "inference_compressed"
    inference_compressed_dir.mkdir(exist_ok=True)
    created_datasets = []

    dataset_paths = sorted(inference_datasets.glob("*.zarr"), key=lambda p: p.name)
    if not dataset_paths:
        print(f"No datasets found in {inference_datasets}")
    else:
        print(f"Found {len(dataset_paths)} datasets. Processing with 24 workers...")

    with ProcessPoolExecutor(max_workers=24) as executor:
        future_to_paths = {}
        for dataset_path in dataset_paths:
            output_path = inference_compressed_dir / dataset_path.name
            future = executor.submit(_compress_one, str(dataset_path), str(output_path))
            future_to_paths[future] = (dataset_path, output_path)

        for future in as_completed(future_to_paths):
            dataset_path, output_path = future_to_paths[future]
            try:
                stats = future.result()
                print(f"Finished processing {dataset_path.name}: {stats}")
                created_datasets.append(output_path)
            except Exception as e:
                print(f"Failed processing {dataset_path.name}: {e}")
