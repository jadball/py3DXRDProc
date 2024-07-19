# py3DXRDProc - Python 3DXRD Processing Toolkit - Diamond Light Source and
# University of Birmingham.
#
# Copyright (C) 2019-2024  James Ball
# Copyright (C) 2005-2019  Jon Wright
#
# This file is part of py3DXRDProc.
#
# py3DXRDProc is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# py3DXRDProc is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with py3DXRDProc. If not, see <https://www.gnu.org/licenses/>.

# Modified from ImageD11/ImageD11/peaksearcher.py at https://github.com/FABLE-3DXRD/ImageD11/

import os
import queue
import sys
import time

import fabio
import numpy as np
from ImageD11 import blobcorrector, ImageD11_thread, ImageD11options
from ImageD11.labelimage import labelimage
from fabio.openimage import openimage
from py3DXRDProc.Indexing.custom_corrector import correct


class timer:
    def __init__(self):
        self.start = time.time()
        self.now = self.start
        self.msgs = []

    def msg(self, msg):
        self.msgs.append(msg)

    def tick(self, msg=""):
        now = time.time()
        self.msgs.append("%s %.2f/s " % (msg, now - self.now))
        self.now = now

    def tock(self, msg=""):
        self.tick(msg)
        print(" ".join(self.msgs), "%.2f/s " % (self.now - self.start))
        sys.stdout.flush()


def peaksearch(filename,
               data_object,
               corrector,
               thresholds,
               labims):
    """
    filename  : The name of the image file for progress info
    data_object : Fabio object containing data and header
    corrector : spatial and dark, linearity etc

    thresholds : [ float[1], float[2] etc]

    labims : label image objects, one for each threshold

    """
    t = timer()
    picture = data_object.data.astype(np.float32)

    assert "Omega" in data_object.header, "Bug in peaksearch headers"

    for lio in list(labims.values()):
        f = lio.sptfile
        f.write("\n\n# File %s\n" % (filename))
        f.write("# Frame %d\n" % (data_object.currentframe))
        f.write("# Processed on %s\n" % (time.asctime()))
        try:
            f.write("# Spatial correction from %s\n" % (corrector.splinefile))
            f.write("# SPLINE X-PIXEL-SIZE %s\n" % (str(corrector.xsize)))
            f.write("# SPLINE Y-PIXEL-SIZE %s\n" % (str(corrector.ysize)))
        except:
            pass
        for item in list(data_object.header.keys()):
            if item == "headerstring":  # skip
                continue
            try:
                f.write("# %s = %s\n" % (item,
                                         str(data_object.header[item]).replace("\n", " ")))
            except KeyError:
                pass

    # Get the rotation angle for this image
    ome = float(data_object.header["Omega"])
    # print "Reading from header"
    #
    # Now peaksearch at each threshold level
    t.tick(filename)
    for threshold in thresholds:
        labelim = labims[threshold]
        f = labelim.sptfile
        if labelim.shape != picture.shape:
            raise "Incompatible blobimage buffer for file %s" % (filename)
        #
        #
        # Do the peaksearch
        f.write("# Omega = %f\n" % (ome))
        labelim.peaksearch(picture, threshold, ome)
        f.write("# Threshold = %f\n" % (threshold))
        f.write("# npks = %d\n" % (labelim.npk))
        #
        if labelim.npk > 0:
            labelim.output2dpeaks(f)
        labelim.mergelast()
        t.msg("T=%-5d n=%-5d;" % (int(threshold), labelim.npk))
        # Close the output file
    # Finish progress indicator for this file
    t.tock()
    sys.stdout.flush()
    return None


def simple_peaksearcher(options, fso):
    corrfunc = blobcorrector.perfect()

    # This is always the case now
    corrfunc.orientation = "edf"

    file_series_object = fso

    if options.out_file[-4:] != ".spt":
        options.out_file = options.out_file + ".spt"
        print("Your output file must end with .spt, changing to ", options.out_file)

    # Make a blobimage the same size as the first image to process

    # List comprehension - convert remaining args to floats
    # must be unique list so go via a set
    thresholds_list = list(set([float(t) for t in options.thresholds]))
    thresholds_list.sort()

    li_objs = {}  # label image objects, dict of

    first_image = next(file_series_object)

    s = first_image.data.shape  # data array shape

    # Create label images
    for t in thresholds_list:
        # the last 4 chars are guaranteed to be .spt above
        mergefile = "%s_t%d.flt" % (options.out_file[:-4], t)
        spotfile = "%s_t%d.spt" % (options.out_file[:-4], t)
        li_objs[t] = labelimage(shape=s,
                                fileout=mergefile,
                                spatial=corrfunc,
                                sptfile=spotfile)
        print("make labelimage", mergefile, spotfile)
    if options.dark is not None:
        print("Using dark (background)", options.dark)
        darkimage = openimage(options.dark).data.astype(np.float32)
    else:
        darkimage = None

    start = time.time()
    print("File being treated in -> out, elapsed time")
    # If we want to do read-ahead threading we fill up a Queue object with data
    # objects
    # THERE MUST BE ONLY ONE peaksearching thread for 3D merging to work
    # there could be several read_and_correct threads, but they'll have to get the order right,
    # for now only one

    print("Going to use threaded version!")
    try:
        # TODO move this to a module ?

        class read_only(ImageD11_thread.ImageD11_thread):
            def __init__(self, queue, file_series_obj, myname="read_only"):
                """ Reads files in file_series_obj, writes to queue """
                self.queue = queue
                self.file_series_obj = file_series_obj
                ImageD11_thread.ImageD11_thread.__init__(self,
                                                         myname=myname)
                print("Reading thread initialised", end=' ')

            def ImageD11_run(self):
                """ Read images and copy them to self.queue """
                for inc, data_object in enumerate(self.file_series_obj):
                    if self.ImageD11_stop_now():
                        print("Reader thread stopping")
                        break
                    if not isinstance(data_object, fabio.fabioimage.fabioimage):
                        # Is usually an IOError
                        if isinstance(data_object[1], IOError):
                            #                                print data_object
                            #                                print data_object[1]
                            sys.stdout.write(str(data_object[1].strerror) + '\n')
                        #                                  ': ' + data_object[1].filename + '\n')
                        else:
                            import traceback
                            traceback.print_exception(data_object[0], data_object[1], data_object[2])
                            sys.exit()
                        continue
                    ti = timer()
                    filein = data_object.filename + "[%d]" % data_object.currentframe + "[%.3f]" % data_object.header["dset_omega"]
                    data_object.header["Omega"] = float(data_object.header["dset_omega"])
                    ti.tick(filein)
                    self.queue.put((filein, data_object), block=True)
                    ti.tock(" enqueue ")
                    if self.ImageD11_stop_now():
                        print("Reader thread stopping")
                        break

                # Flag the end of the series
                self.queue.put((None, None), block=True)

        class correct_one_to_many(ImageD11_thread.ImageD11_thread):
            def __init__(self, queue_read, queues_out, thresholds_list,
                         dark=None, myname="correct_one",
                         monitorcol=None, monitorval=None):
                """ Using a single reading queue retains a global ordering
                corrects and copies images to output queues doing
                correction once """
                self.queue_read = queue_read
                self.queues_out = queues_out
                self.dark = dark
                self.monitorcol = monitorcol
                self.monitorval = monitorval
                self.thresholds_list = thresholds_list

                ImageD11_thread.ImageD11_thread.__init__(self,
                                                         myname=myname)

            def ImageD11_run(self):
                while not self.ImageD11_stop_now():
                    ti = timer()
                    filein, data_object = self.queue_read.get(block=True)
                    if filein is None:
                        for t in self.thresholds_list:
                            self.queues_out[t].put((None, None),
                                                   block=True)
                        # exit the while 1
                        break
                    data_object = correct(data_object, self.dark,
                                          monitorval=self.monitorval,
                                          monitorcol=self.monitorcol,
                                          )
                    ti.tick(filein + " correct ")
                    for t in self.thresholds_list:
                        # Hope that data object is read only
                        self.queues_out[t].put((filein, data_object),
                                               block=True)
                    ti.tock(" enqueue ")
                print("Corrector thread stopping")

        class peaksearch_one(ImageD11_thread.ImageD11_thread):
            def __init__(self, q, corrfunc, threshold, li_obj,
                         myname="peaksearch_one"):
                """ This will handle a single threshold and labelimage
                object """
                self.q = q
                self.corrfunc = corrfunc
                self.threshold = threshold
                self.li_obj = li_obj
                ImageD11_thread.ImageD11_thread.__init__(
                    self,
                    myname=myname + "_" + str(threshold))

            def run(self):
                while not self.ImageD11_stop_now():
                    filein, data_object = self.q.get(block=True)
                    if not isinstance(data_object, fabio.fabioimage.fabioimage):
                        break
                    peaksearch(filein, data_object, self.corrfunc,
                               [self.threshold],
                               {self.threshold: self.li_obj})
                self.li_obj.finalise()

        # 8 MB images - max 40 MB in this queue
        read_queue = queue.Queue(5)
        reader = read_only(read_queue, file_series_object)
        reader.start()
        queues = {}
        searchers = {}
        for t in thresholds_list:
            print("make queue and peaksearch for threshold", t)
            queues[t] = queue.Queue(3)
            searchers[t] = peaksearch_one(queues[t], corrfunc, t, li_objs[t])
        corrector = correct_one_to_many(read_queue,
                                        queues,
                                        thresholds_list,
                                        dark=darkimage,
                                        monitorcol=options.monitor_col,
                                        monitorval=options.monitor_val,
                                        )
        corrector.start()
        my_threads = [reader, corrector]
        for t in thresholds_list[::-1]:
            searchers[t].start()
            my_threads.append(searchers[t])
        nalive = len(my_threads)

        def empty_queue(q):
            while 1:
                try:
                    q.get(block=False, timeout=1)
                except:
                    break
            q.put((None, None), block=False)

        while nalive > 0:
            try:
                nalive = 0
                for thr in my_threads:
                    if thr.is_alive():
                        nalive += 1
                if options.kill_file is not None and \
                        os.path.exists(options.kill_file):
                    raise KeyboardInterrupt()
                time.sleep(1)
            except KeyboardInterrupt:
                print("Got keyboard interrupt in waiting loop")
                ImageD11_thread.stop_now = True
                try:
                    time.sleep(1)
                except:
                    pass
                empty_queue(read_queue)
                for t in thresholds_list:
                    q = queues[t]
                    empty_queue(q)
                for thr in my_threads:
                    if thr.is_alive():
                        thr.join(timeout=1)
                print("finishing from waiting loop")
            except:
                print("Caught exception in waiting loop")
                ImageD11_thread.stop_now = True
                time.sleep(1)
                empty_queue(read_queue)
                for t in thresholds_list:
                    q = queues[t]
                    empty_queue(q)
                for thr in my_threads:
                    if thr.is_alive():
                        thr.join(timeout=1)
                raise

    except ImportError:
        print("Probably no threading module present")
        raise


def get_options(parser):
    """ Add our options to a parser object """
    parser.add_argument("-o", "--out_file", action="store",
                        dest="out_file", default="peaks.spt", type=str,
                        help="Output filename, default=peaks.spt")
    parser.add_argument("-d", "--dark_file", action="store",
                        dest="dark", default=None, type=ImageD11options.ImageFileType(mode='r'),
                        help="Dark current filename, to be subtracted, default=None")
    parser.add_argument("-t", "--threshold", action="append", type=float,
                        dest="thresholds", default=None,
                        help="Threshold level, you can have several")
    parser.add_argument("-k", "--kill_file", action="store",
                        dest="kill_file", default=None,
                        type=ImageD11options.FileType(),
                        help="Name of file to create stop the peaksearcher running")
    parser.add_argument("--monitor_col", action="store", type=str,
                        dest="monitor_col",
                        default=None,
                        help="Header value for incident beam intensity")
    parser.add_argument("--monitor_val", action="store", type=float,
                        dest="monitor_val",
                        default=None,
                        help="Incident beam intensity value to normalise to")

    return parser
