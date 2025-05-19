from setuptools import setup, find_packages

requirements = [
    'gymnasium',
    # 'gymnasium[classic-control]'
]

train_requirements = [
    'stable-baselines3',
    'tensorboard',
    'wandb',
    'moviepy',
]

setup(
    name="AwesomeProject",
    version="0.1.0",
    author="Phil McCavity",
    author_email="phil.mccavity@example.com",
    description="A completely working example",
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
