#! /usr/bin/env python3
"""
Created on 6 avr. 2023

@author: jcolley
"""


import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pprint

import rshower.manage_log as mlg
from rshower.io.shower.zhaires_master import ZhairesMaster


# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("error", log_stdout=True)


def manage_args():
    parser = argparse.ArgumentParser(description="Information and plot event/traces")
    parser.add_argument("path", help="path of ZHAireS single event simulation ", type=Path)
    parser.add_argument(
        "-f",
        "--footprint",
        help="interactive plot (double click) of footprint, time max value and value for each station",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--time_val",
        help="interactive plot, value of each station at time t defined by a slider",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--trace",
        help="plot trace x,y,z and power spectrum of detector unit (station)",
        default="",
    )
    parser.add_argument(
        "--trace_image",
        action="store_true",
        required=False,
        help="interactive image plot (double click) of norm of traces",
    )
    parser.add_argument(
        "--list_du",
        action="store_true",
        required=False,
        help="list of identifier of station",
    )
    parser.add_argument(
        "--dump",
        default="",
        help="dump trace of station",
    )  # retrieve argument
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        required=False,
        help="some information about the contents of the file",
    )  # retrieve argument
    return parser.parse_args()


def main():
    #
    args = manage_args()
    d_event = ZhairesMaster(str(args.path))
    o_tevent = d_event.get_object_3dtraces()
    if args.info:
        print(f"Nb station         : {o_tevent.get_nb_du()}")
        print(f"Size trace    : {o_tevent.get_size_trace()}")
        pprint.pprint(d_event.get_simu_info())
    if args.list_du:
        print(f"\nIdentifier station : ")
        s_id = ""
        for id_du in o_tevent.idt2idx.keys():
            s_id += f" {id_du} ," 
        print(s_id[1:-1]) 
    if args.trace_image:
        o_tevent.plot_all_traces_as_image()
    if args.footprint:
        o_tevent.plot_footprint_val_max()
        o_tevent.plot_footprint_4d_max()
    if args.time_val:
        o_tevent.plot_footprint_time_max()
    if args.trace != "":
        if not args.trace in o_tevent.idt2idx.keys():
            logger.error(f"ERROR: unknown station identifer")
            return
        o_tevent.plot_trace_du(args.trace)
        o_tevent.plot_ps_trace_du(args.trace)
    if args.dump != "":
        if not args.dump in o_tevent.idt2idx.keys():
            logger.error(f"ERROR: unknown station identifer")
            return
        idx_du = o_tevent.idt2idx[args.dump]
        tr_du = o_tevent.traces[idx_du]
        t_tr = o_tevent.t_samples[idx_du]
        for idx in range(o_tevent.get_size_trace()):
            print(f"{t_tr[idx]} {tr_du[0,idx]} {tr_du[1,idx]} {tr_du[2,idx]}")


if __name__ == "__main__":
    main()
    plt.show()
