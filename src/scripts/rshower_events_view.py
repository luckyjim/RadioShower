#! /usr/bin/env python3
"""
Created on 22 August 2024

@author: jcolley
"""


import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pprint

import rshower.manage_log as mlg
from rshower.io.events.asdf_traces import AsdfReadTraces


# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("error", log_stdout=True)


def manage_args():
    parser = argparse.ArgumentParser(description="Muliti events viewer from GRAND network GP13")
    parser.add_argument("file", help="path and name of file GRAND", type=argparse.FileType("r"))
    parser.add_argument(
        "-f",
        "--footprint",
        help="interactive plot (double click) of footprint, max value for each DU",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--time_val",
        help="interactive plot, value of each DU at time t defined by a slider",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--trace",
        type=int,
        help="plot trace x,y,z and power spectrum of detector unit (DU)",
        default=-100,
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help="Select event with index <index>, given by -i option, index is always > 0 or = 0",
        default=-100,
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
        help="list of identifier of DU",
    )
    parser.add_argument(
        "--dump",
        type=int,
        default=-100,
        help="dump trace of DU",
    )  # retrieve argument
    parser.add_argument(
        "--info",
        action="store_true",
        required=False,
        help="Some information and plots about the contents of the file",
    )  # retrieve argument
    return parser.parse_args()


def main():
    #
    logger.info("Example script to deal with 3D traces.")

    args = manage_args()
    d_events = AsdfReadTraces(args.file.name)
    # d_events = GrandEventsSelectedFmt01(args.file.name)
    if args.index != -100:
        if args.index < 0:
            logger.error("index events must >= 0")
            return
        if args.index >= d_events.nb_events:
            logger.error(f"index events must < {d_events.nb_events}")
            return
        o_tevent = d_events.get_event(args.index)
        # o_tevent.set_noise_interval(624,1024)
    if args.info:
        d_events.plot_hist_du()
        d_events.plot_hist_xcore()
        d_events.plot_hist_xmax()
        idx_du, nb_du_in_event = d_events.get_nb_du_in_event_sort()
        print("Idx event , Nb du in event")
        for idx, nb_du in zip(idx_du, nb_du_in_event):
            print(f"{idx}, {nb_du}")
    if args.list_du:
        print(f"Identifier DU : {o_tevent.idx2idt}")
    # if args.trace_image:
    #     o_tevent.plot_all_traces_as_image()
    if args.footprint:
        o_tevent.plot_footprint_4d_max()
        o_tevent.plot_footprint_val_max()
    if args.time_val:
        o_tevent.plot_footprint_time_slider()
    if args.trace != -100:
        if not args.trace in o_tevent.idt2idx.keys():
            logger.error(f"ERROR: unknown DU identifer")
            return
        o_tevent.plot_trace_du(args.trace)
        o_tevent.plot_psd_trace_du(args.trace)
    if args.dump != -100:
        if not args.dump in o_tevent.idt2idx.keys():
            logger.error(f"ERROR: unknown DU identifer")
            return
        idx_du = o_tevent.idt2idx[args.dump]
        tr_du = o_tevent.traces[idx_du]
        t_tr = o_tevent.t_samples[idx_du]
        for idx in range(o_tevent.get_size_trace()):
            print(f"{t_tr[idx]} {tr_du[0,idx]} {tr_du[1,idx]} {tr_du[2,idx]}")


if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # =============================================
    main()
    # =============================================
    plt.show()
    logger.info(mlg.string_end_script())
