# RadioShower

** Prelimary version **

** Work in progress **

## Installation

```bash
$ pip install git+https://github.com/luckyjim/RadioShower.git
```

## Python library

```python
import rshower
```

## Script 

```bash
$ zhaires_view.py -h
usage: zhaires_view.py [-h] [-f] [--time_val] [-t TRACE] [--trace_image] [--list_du] [--dump DUMP] [-i] path

Information and plot event/traces for ROOT file

positional arguments:
  path                  path of ZHAireS single event simulation

options:
  -h, --help            show this help message and exit
  -f, --footprint       interactive plot (double click) of footprint, time max value and value for each DU
  --time_val            interactive plot, value of each DU at time t defined by a slider
  -t TRACE, --trace TRACE
                        plot trace x,y,z and power spectrum of detector unit (DU)
  --trace_image         interactive image plot (double click) of norm of traces
  --list_du             list of identifier of DU
  --dump DUMP           dump trace of DU
  -i, --info            some information about the contents of the file
```
