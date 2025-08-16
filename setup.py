from setuptools import setup, find_packages

setup(
    name='chess_bot',
    version='0.1.0',
    description='A short description of your project',
    author='Lars Leijten',
    author_email='lars.leijten@radboudumc.nl',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'chess',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)