
from diffusion_policy_contrib.real_world.single_realsense import SingleRealsense
from diffusion_policy_contrib.shared_memory.shared_memory_util import SharedMemoryManager
import cv2
import numpy as np


if __name__ == "__main__":
    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
                    serial_number="938422072678",
                    shm_manager=shm_manager,
                    resolution=(320, 240),
                    capture_fps=30,
                    put_fps=30,
                    put_downsample=False,
                    verbose=False,
            ) as cam:

            while True:
                image_data = cam.get()

                if isinstance(image_data, dict) and "color" in image_data:
                    rgb_img = image_data["color"]
                elif isinstance(image_data, dict) and len(image_data) > 0:
                    first_val = next(iter(image_data.values()))
                    if isinstance(first_val, dict) and "color" in first_val:
                        rgb_img = first_val["color"]
                    else:
                        raise ValueError("SingleRealsense output does not include a 'color' frame")
                else:
                    raise ValueError("Unexpected SingleRealsense output format")


                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                rgb_img_resized = cv2.resize(rgb_img, (144, 144), interpolation=cv2.INTER_AREA)
                # h, w = rgb_img.shape[:2]
                # crop_size = min(144, h, w)
                # y0 = (h - crop_size) // 2
                # x0 = (w - crop_size) // 2
                # rgb_img = rgb_img[y0 : y0 + crop_size, x0 : x0 + crop_size]
                img_resnet_format = np.moveaxis(rgb_img_resized.astype(np.float32) / 255.0, -1, 0)
                img_cv_format = np.moveaxis(rgb_img_resized, -1, 0)
                cv2.imshow("img", cv2.cvtColor(rgb_img_resized, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)