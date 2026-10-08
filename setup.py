from setuptools import find_packages, setup

setup(
    name="improved-diffusion",
    version="0.1.0",
    description="SNAPHU-conditioned diffusion for InSAR phase unwrapping",
    packages=find_packages(include=["improved_diffusion", "improved_diffusion.*"]),
    license_files=["THIRD_PARTY_NOTICES"],
    install_requires=[
        "blobfile>=1.0.5",
        "torch",
        "numpy",
        "tifffile",
        "matplotlib",
        "scipy",
        "scikit-image",
        "cmcrameri",
        "tqdm",
    ],
    extras_require={
        "mpi": ["mpi4py"],
    },
)
