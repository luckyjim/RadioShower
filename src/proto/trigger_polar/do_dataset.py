"""
Created on 16 oct. 2025

@author: jcolley
"""

import pathlib
import re

from proto.trigger_polar.model_dirvolt import DirectionVoltageParameters


out_dir = "/home/jcolley/projet/lucky/data/v2"

pattern = re.compile(r"^volt")

rep = pathlib.Path(out_dir)

for m_f in rep.iterdir():
    print(m_f)
    if m_f.is_file() and pattern.search(m_f.name):
        f_volt = str(m_f.absolute())
        dirv = DirectionVoltageParameters(f_volt)
        dirv.process_events(0, -1)
