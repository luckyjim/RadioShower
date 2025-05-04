from setuptools import setup
import sys

sys.path.append('src')

import rshower
from pip._internal.req import parse_requirements

# requirements = parse_requirements("shower_radio/requirements_novers.txt", session="hack")
# requires = [str(item.req) for item in requirements]
# print (requires)

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

#print (parse_requirements("shower_radio/requirements_novers.txt"))
my_requires=[
    'requests',
    'importlib-metadata; python_version<"3.10"',
],
my_requires.append(parse_requirements("requirements_novers.txt"))


setup(
    name="RadioShower",
    description="Tools for radio traces from air shower",
    version=rshower.__version__,
    author=rshower.__author__,
    classifiers=[
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/grand-mother/NUTRIG1",
    package_dir={"rshower": "src/rshower"},
    scripts=["src/scripts/zhaires_view.py",
             "src/scripts/grand_events_view.py",
             "src/scripts/plot_tmax_vmax.py"],
    license='MIT', 
    python_requires='>=3.4', 
    #install_requires=["numpy","scipy","matplotlib","asdf","h5py"]
    install_requires=my_requires
)
