# Code quality tools for developper


## Install quality tools


```console
source RadioShower/setup.sh
```
 

```console
python -m pip install -r quality/requirements.txt
```

## Check code, static analysis

We use [pylint](https://www.pylint.org/) as static code analysis to check coding standard (PEP8) and as error detector.

### Configuration 

With file 

```console
quality/pylint.conf
```
main options:

- 'disable' a rule
- 'enable' a rule
- 'ignored-classes' for false positive, for example astropy.units

### Script for rshower

```console
rshower_quality_analysis.bash
```


You can see the module path and line number where pylint detected a problem and then a description of the problem

## Check test coverage : 

We use [coverage.py](https://coverage.readthedocs.io/en/stable/) for measuring code coverage of Python programs. It monitors your program, noting which parts of the code have been executed, then analyzes the source to identify code that could have been executed but was not.

### Configuration 

We use this simple option 

```console
coverage run --source=src/rshower -m pytest tests 
```

### Script for rshower

```console
rshower_quality_test_cov.bash
```


coverage.py provides also pretty HTML ouput page by module that indicate zone coverage. 
Open file 

```console
quality/html_coverage/index.html
```
with a web navigator.

### Launch only one test

During a debugging phase, it is practical to launch only the test you are working on, this is possible with the following command

```console
pytest tests/basis/test_traces_event::test_to_fix
```

At the end of development re-launched

```console
rshower_quality_test_cov.bash
```
to update coverage with correction.




### Configuration 

We use this simple option 

```console
--config-file=tests/mypy.ini 
```

### Script for rshower

```console
rshower_quality_type.bash
```

### Output example


See report in file 

```console
quality/report_type.txt
```

or HTML page here

```console
quality/html_mypy/index.html
```

with web navigator.

