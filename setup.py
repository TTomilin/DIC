from setuptools import setup, find_packages

requirements = [
    'gymnasium',
    'gymnasium[classic-control]',
    'gymnasium[box2d]',
    'levdoom',
]

train_requirements = [
    'stable-baselines3',
    'tensorboard',
    'wandb',
    'moviepy',
]

setup(
    name="Experimental Setup",
    version="0.1.0",
    author="Tristan Tomilin",
    author_email="t.tomilin@tue.nl",
    description="Demo",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'train': train_requirements,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
