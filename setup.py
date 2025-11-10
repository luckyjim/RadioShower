from setuptools import setup, find_packages
import os

# Lecture de la version sans importer le module
def read_version():
    version_file = os.path.join("src_lib", "rshower", "__init__.py")
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.0.0"

my_requires = [
    "requests",
    'importlib-metadata; python_version<"3.10"',
    "asdf",
    "h5py",
    "matplotlib",
    "numpy",
    "scipy",
]

setup(
    name="RadioShower",
    version=read_version(),
    description="Tools for radio traces from air shower",
    author="Lucky Jim",  # ou rshower.__author__ si tu veux le garder
    license="MIT",
    url="https://github.com/luckyjim/RadioShower",
    python_requires=">=3.4",
    install_requires=my_requires,
    packages=find_packages(where="src_lib"),
    package_dir={"": "src_lib"},
    classifiers=[
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    scripts=["src/scripts/zhaires_view.py",
             "src/scripts/rshower_events_view.py",
             "src/scripts/grand_events_view.py",
             "src/scripts/plot_tmax_vmax.py"],
)