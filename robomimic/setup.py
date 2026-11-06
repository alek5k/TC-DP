from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
readme_path = path.join(this_directory, 'README.md')
if not path.exists(readme_path):
    readme_path = path.join(this_directory, 'README_ROBOMIMIC.md')
with open(readme_path, encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if (('.png' not in x) and ('.gif' not in x))]
long_description = ''.join(lines)

setup(
    name="robomimic",
    packages=[
        package for package in find_packages() if package.startswith("robomimic")
    ],
    install_requires=[
        "numpy>=1.13.3",
        "h5py",
        "psutil",
        "tqdm",
        "termcolor",
        "tensorboard",
        "tensorboardX",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "egl_probe>=1.0.1",
        "torch",
        "torchvision",
        "huggingface_hub==0.23.4",
        "transformers==4.41.2",
        "diffusers==0.11.1",
    ],
    extras_require={
        "temporal-envs": [
            "gym==0.26.2",
            "pygame==2.1.2",
            "pymunk==6.2.1",
            "shapely==1.8.4",
            "opencv-python>=4.6",
            "scipy>=1.9",
            "scikit-image>=0.19",
            "zarr<3",
            "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@95743f6687ad7394ccb6e865fa4e1b99114e580d",
        ],
    },
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="robomimic: A Modular Framework for Robot Learning from Demonstration",
    author="Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Matthew Bronars, Vaibhav Saxena",
    url="https://github.com/ARISE-Initiative/robomimic",
    author_email="amandlek@cs.stanford.edu",
    version="0.5.0",
    long_description=long_description,
    long_description_content_type='text/markdown'
)
