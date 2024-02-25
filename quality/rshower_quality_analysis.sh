#!/bin/bash

cd $RSHOWER_ROOT
# add -r y to add various information, statistics, ...
pylint src/rshower --rcfile quality/pylint.conf --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" --output=quality/report_pylint.txt
status=$?
cat quality/report_pylint.txt
exit $status