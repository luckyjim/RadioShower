from setuptools import setup
import sys

sys.path.append('src')

import rshower

my_requires=[
    'requests',
    'importlib-metadata; python_version<"3.10"',
    'asdf',
    'h5py',
    'matplotlib',
    'numpy',
    'scipy',
]


setup(
    name="RadioShower",
    description="Tools for radio traces from air shower",
    version=rshower.__version__,
    author=rshower.__author__,
    classifiers=[
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/luckyjim/RadioShower",
    package_dir={"rshower": "src/rshower"},
    scripts=["src/scripts/zhaires_view.py",
             "src/scripts/grand_events_view.py",
             "src/scripts/plot_tmax_vmax.py"],
    license='MIT', 
    python_requires='>=3.4',
    install_requires=my_requires
)
