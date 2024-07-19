#  py3DXRDProc - Python 3DXRD Processing Toolkit - Diamond Light Source and
#  University of Birmingham.
#
#  Copyright (C) 2019-2024  James Ball unless otherwise stated
#
#  This file is part of py3DXRDProc.
#
#  py3DXRDProc is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the
#  Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  py3DXRDProc is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with py3DXRDProc. If not, see <https://www.gnu.org/licenses/>.

import argparse
import os
import re
import subprocess
from typing import Tuple

from py3DXRDProc.parse_input_file import parameter_attribute_builder, Struct


def dls_correct_environment():
    """Makes sure that DLS modules exist in the path"""
    # Created by Bill Pulford?
    # DLS Confluence/Computing Technical Documentation/Frequently Asked Questions
    # Adaptation of /usr/share/Modules/init/python.py
    if "MODULEPATH" not in os.environ:
        f = open(os.environ["MODULESHOME"] + "/init/.modulespath", "r")
        path = []
        for line in f.readlines():
            line = re.sub("#.*$", "", line)
            if line != "":
                path.append(line)
        os.environ["MODULEPATH"] = ":".join(path)

    if "LOADEDMODULES" not in os.environ:
        os.environ["LOADEDMODULES"] = ""


def dls_module(*supplied_args):
    """Python version of DLS `module load x` command"""
    # Created by Bill Pulford?
    # DLS Confluence/Computing Technical Documentation/Frequently Asked Questions
    # Adaptation of /usr/share/Modules/init/python.py
    if type(supplied_args[0]) is list:
        supplied_args = supplied_args[0]
    else:
        supplied_args = list(supplied_args)
    (output, error) = subprocess.Popen(
        ["/usr/bin/modulecmd", "python"] + supplied_args, stdout=subprocess.PIPE
    ).communicate()
    exec(output)


def dls_submit_cluster_job_simple(sh_script_path: str, n_tasks: int, index_files_dir: str, simul_job_lim: int):
    """Submit cluster array job to Diamond Cluster

    :param sh_script_path: Path to shell script you want to submit
    :param n_tasks: Number of array tasks to submit
    :param index_files_dir: Path to directory containing numbered .inp files, one per job
    :param simul_job_lim: Max number of allowed simultaneous jobs
    """

    from subprocess import call
    dls_correct_environment()
    dls_module("load", "global/cluster")
    job_submission_script_string = f"qsub -t 1-{n_tasks} -tc {simul_job_lim} {sh_script_path} {index_files_dir}"
    call(['/bin/bash', '-i', '-c', job_submission_script_string])


def dls_submit_cluster_job_slurm(sh_script_path: str, n_tasks: int, index_files_dir: str, simul_job_lim: int,
                                 cluster_output_dir: str):
    """Submit cluster array job to new Diamond Slurm Cluster

    :param sh_script_path: Path to shell script you want to submit
    :param n_tasks: Number of array tasks to submit
    :param index_files_dir: Path to directory containing numbered .inp files, one per job
    :param simul_job_lim: Max number of allowed simultaneous jobs
    """

    # make a temporary JSON

    import json

    dispatcher_output_path = os.path.join(cluster_output_dir, "dispatcher_output.out")

    json_data_dict = {"job": {
        "partition": "cs04r",
        "tasks": 1,
        "name": "3DXRD_dispatcher",
        "nodes": 1,
        "current_working_directory": "/tmp",
        "environment": {
            "PATH": "/bin:/usr/bin/:/usr/local/bin/",
            "LD_LIBRARY_PATH": "/lib/:/lib64/:/usr/local/lib"
        },
        "standard_output": dispatcher_output_path
    },
        "script": f"#!/bin/bash\nsbatch --array=1-{n_tasks}%{simul_job_lim} --partition=cs04r {sh_script_path} {index_files_dir}"
    }

    json_object = json.dumps(json_data_dict, indent=4)
    with open("slurm_rest_api_submission.json", "w") as outfile:
        outfile.write(json_object)

    # submit a request with the JSON

    import requests
    payload = open("slurm_rest_api_submission.json")
    headers = {'X-SLURM-USER-NAME': os.environ["USER"],
               'X-SLURM-USER-TOKEN': os.environ["SLURM_JWT"],
               'Content-Type': "application/json"
               }
    url = os.environ["SLURM_URL"]
    r = requests.post(url, data=payload, headers=headers)
    print(r)


def esrf_submit_cluster_job_slurm(sh_script_path: str, n_tasks: int, index_files_dir: str, simul_job_lim: int,
                                  cluster_output_dir: str):
    """Submit cluster array job to ESRF Slurm Cluster
 
     :param sh_script_path: Path to shell script you want to submit
     :param n_tasks: Number of array tasks to submit
     :param index_files_dir: Path to directory containing numbered .inp files, one per job
     :param simul_job_lim: Max number of allowed simultaneous jobs
     """

    # make a temporary JSON

    import json

    dispatcher_output_path = os.path.join(cluster_output_dir, "dispatcher_output.out")

    json_data_dict = {"job": {
        "partition": "low",
        "tasks": 1,
        "name": "3DXRD_dispatcher",
        "nodes": 1,
        "current_working_directory": "/tmp",
        "environment": {
            "PATH": "/bin:/usr/bin/:/usr/local/bin/",
            "LD_LIBRARY_PATH": "/lib/:/lib64/:/usr/local/lib"
        },
        "standard_output": dispatcher_output_path
    },
        "script": f"#!/bin/bash\nsbatch --array=1-{n_tasks}%{simul_job_lim} --partition=low {sh_script_path} {index_files_dir}"
    }

    json_object = json.dumps(json_data_dict, indent=4)
    with open("slurm_rest_api_submission.json", "w") as outfile:
        outfile.write(json_object)

    # submit a request with the JSON

    import requests
    payload = open("slurm_rest_api_submission.json")
    headers = {'X-SLURM-USER-NAME': os.environ["USER"],
               'X-SLURM-USER-TOKEN': os.environ["SLURM_JWT"],
               'Content-Type': "application/json"
               }
    url = os.environ["SLURM_URL"]
    r = requests.post(url, data=payload, headers=headers)
    print(r)


def bham_submit_cluster_job(sh_script_path: str, n_tasks: int, index_files_dir: str, simul_job_lim: int):
    """Submit cluster array job to BEAR Cluster

    :param sh_script_path: Path to shell script you want to submit
    :param n_tasks: Number of array tasks to submit
    :param index_files_dir: Path to directory containing numbered .inp files, one per job
    :param simul_job_lim: Max number of allowed simultaneous jobs
    """
    from subprocess import call
    job_submission_script_string = f"sbatch --array=1-{n_tasks}%{simul_job_lim} {sh_script_path} {index_files_dir}"
    call(['/bin/bash', '-i', '-c', job_submission_script_string])


def write_array_file(load_step: str, scan: int, options: argparse.Namespace, index_files_dir: str, index: int,
                     load_step_path: str, sample_name: str, key: str = None):
    """Write an individual array file that will get read by the job submission script

    :param load_step: Load step we are indexing
    :param scan: Scan number we are indexing
    :param options: Command-line options from index_3DXRD.py
    :param index_files_dir: Path to directory containing numbered .inp files, one per job
    :param index: Specific index for this file
    :param load_step_path: Path to load step processing directory
    :param sample_name: Name of sample
    :param key: HDF5 key of scan (only relevant for ESRF data), defaults to None
    """
    destination_file_path = os.path.join(index_files_dir, str(index) + ".inp")
    input_file_real_path = os.path.abspath(options.input_file)
    lines_to_write = [
        input_file_real_path + "\n",
        load_step + "\n",
        load_step_path + "\n",
        str(scan) + "\n",
        sample_name + "\n"
    ]
    if key is not None:
        lines_to_write.append(key + "\n")
    with open(destination_file_path, "w") as dest:
        dest.writelines(lines_to_write)


def read_array_file(inp_file_path: str) -> Tuple[Struct, str, str, str, str, str]:
    """Read an individual array file written by write_array_file

    `pars: The :class:`Struct` containing the indexing parameters

    `load_step`: The name of the load step

    `load_step_processing_dir`: The path to the load step subdirectory in the processing directory

    `scan_name`: The name of the scan

    `sample_name`: The name of the sample

    `key`: The optional name of the HDF5 key containing the fly scan (ESRF only)

    :param inp_file_path: Path to individual input file
    :return: A tuple of `(pars Struct, load step name, load step processing directory, scan name, sample name, key)`
    """

    key = None
    with open(inp_file_path, "r") as inp_file:
        inp_file_lines = inp_file.readlines()

    inp_file_lines_stripped = [x.replace("\n", "") for x in inp_file_lines]
    pars_path = inp_file_lines_stripped[0]

    pars = parameter_attribute_builder(pars_path)

    load_step = inp_file_lines_stripped[1]
    load_step_processing_dir = inp_file_lines_stripped[2]
    scan_name = inp_file_lines_stripped[3]
    sample_name = inp_file_lines_stripped[4]
    if pars.collection_facility == "ESRF":
        key = inp_file_lines_stripped[5]

    return pars, load_step, load_step_processing_dir, scan_name, sample_name, key


def get_number_of_cores() -> int:
    """Get the number of CPU cores available on a cluster node.
    Defaults to using `multiprocessing.cpu_count()` if not running on DLS or BEAR clusters.

    :return: The number of CPU cores available
    """
    # Find out how many CPUs we have
    ncpu_str = os.getenv('NSLOTS')
    if ncpu_str is None:
        ncpu_str = os.getenv('SLURM_NTASKS')
        if ncpu_str is None:
            # use multiprocessing value
            import multiprocessing as mp
            ncpu_str = str(mp.cpu_count())

    ncpu = int(ncpu_str)

    return max(min(ncpu, 40), 4)
