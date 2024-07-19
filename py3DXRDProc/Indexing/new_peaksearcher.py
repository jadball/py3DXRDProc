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

# Modified from ImageD11/sandbox/ma4750/custom_peak_search.py at https://github.com/jonwright/ImageD11/

import functools
import multiprocessing
import sys
import timeit

import ImageD11.cImageD11
import fabio
import h5py
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.spatial
import tqdm
import logging
log = logging.getLogger(__name__)



def do3dmerge(honame, scan):
    # read the 2d peak search results
    t0 = timeit.default_timer()
    log.info('Making 3D merge')
    sys.stdout.flush()
    with h5py.File(honame, 'r') as hin:
        g = hin[scan]['peaks2d']
        h5input = g.attrs['h5input']
        h5scan = g.attrs['h5scan']
        detector_path = g.attrs['detector_path']
        omega_path = g.attrs['omega_path']
        omega = g['omega'][:]
        s = g['s_raw'][:]
        f = g['f_raw'][:]
        o = g['o_raw'][:]
        I = g['s_I'][:]
        n = g['npks'][:]
        s1 = g['s_1'][:]
    # pointers to frames in s,f,o,I,n,s1
    p = np.cumsum(np.concatenate(([0, ], n)))
    # make a KDTree for each frame (wastes a bit of memory, but easier for sorting later)
    trees = [scipy.spatial.cKDTree(np.transpose((s[p[i]:p[i + 1]], f[p[i]:p[i + 1]])))
             for i in range(len(n))]
    log.info('made trees')
    sys.stdout.flush()
    # because interlaced might not be in order
    order = np.argsort(omega % 360)
    # peaks that overlap, k : 0 -> npks == len(s|f|o)
    # diagonal
    krow = list(range(len(o)))
    kcol = list(range(len(o)))
    for i in range(1, len(n)):  # match these to previous
        flo = order[i - 1]
        fhi = order[i]
        tlo = trees[flo]
        thi = trees[fhi]
        # 1.6 is how close centers should be to overlap
        lol = trees[flo].query_ball_tree(trees[fhi], r=1.6)
        for srcpk, destpks in enumerate(lol):  # dest is strictly higher than src
            for destpk in destpks:
                krow.append(srcpk + p[flo])
                kcol.append(destpk + p[fhi])
    csr = scipy.sparse.csr_matrix((np.ones(len(krow), dtype=bool),
                                   (kcol, krow)), shape=(len(o), len(o)))
    # connected components == find all overlapping peaks
    ncomp, labels = scipy.sparse.csgraph.connected_components(csr,
                                                              directed=False, return_labels=True)
    log.info('connected components')
    sys.stdout.flush()
    # Now merge the properties
    npkmerged = np.bincount(labels, minlength=ncomp)  # number of peaks that were merged
    s3d1 = np.bincount(labels, minlength=ncomp, weights=s1)  # s_1
    s3dI = np.bincount(labels, minlength=ncomp, weights=I)  # s_I
    ssI = np.bincount(labels, minlength=ncomp, weights=I * s)  # s_sI
    sfI = np.bincount(labels, minlength=ncomp, weights=I * f)  # s_sI
    soI = np.bincount(labels, minlength=ncomp, weights=I * o)  # s_sI
    s3d = ssI / s3dI
    f3d = sfI / s3dI
    o3d = soI / s3dI
    with h5py.File(honame, 'a') as hin:
        g = hin[scan].require_group('peaks3d')
        g['s_raw'] = s3d
        g['f_raw'] = f3d
        g['omega'] = o3d
        g['sum_intensity'] = s3dI
        g['Number_of_pixels'] = s3d1
    t1 = timeit.default_timer()
    log.info('wrote %.3f/s' % (t1 - t0))


class worker:
    """ subtracts background, custom for ma47050 """

    def __init__(self, bgfile):
        self.bg = fabio.open(bgfile).data.astype(np.float32)
        self.threshold = 50  # ADU to zero out image
        self.smoothsigma = 1.  # sigma for Gaussian before labelleing
        self.bgc = 0.9  # fractional part of bg per peak to remove
        self.minpx = 3

        self.m_offset = self.bg < 100
        self.mbg = np.mean(self.bg[self.m_offset])
        self.m_ratio = self.bg > 300
        self.bg -= self.mbg  # remove dark
        self.invbg = 1 / self.bg[self.m_ratio]
        self.bins = b = np.linspace(0, 2, 256)
        self.bc = (b[1:] + b[:-1]) / 2

        self.wrk = np.empty(self.bg.shape, 'b')
        self.labels = np.empty(self.bg.shape, 'i')

    def bgsub(self, img):
        img = img.astype(np.float32)
        offset = np.mean(img[self.m_offset])  # remove dark
        np.subtract(img, offset, img)
        ratio = img[self.m_ratio] * self.invbg
        h, b = np.histogram(ratio, bins=self.bins)
        htrim = np.where(h < h.max() * 0.05, 0, h)
        r = (htrim * self.bc).sum() / htrim.sum()
        # Better to scale background to data rather than data to background?
        # np.multiply(img, 1 / r, img)
        np.subtract(img, self.bg * r, img)
        self.offset = offset
        self.scale = r
        return img

    def peaksearch(self, img, omega=0):
        self.cor = self.bgsub(img)
        # smooth the image for labelling (removes noise maxima)
        self.smoothed = scipy.ndimage.gaussian_filter(self.cor, self.smoothsigma)
        assert self.smoothed.dtype == np.float32
        # zero out the background
        self.mt = self.smoothed < self.threshold
        self.smoothed[self.mt] = 0
        # label on smoothed image
        self.npks = ImageD11.cImageD11.localmaxlabel(self.smoothed, self.labels, self.wrk)
        self.labels[self.mt] = 0
        # now find the borders of each blob : first dilate
        l3 = scipy.ndimage.uniform_filter(self.labels * 7, 3)
        self.borders = (self.labels * 7) != l3
        # border properties - use the real data or the smoothed? Real data. Might be noisier
        self.blobsouter = ImageD11.cImageD11.blobproperties(self.cor, self.labels * self.borders, self.npks)
        # Computed background per peak
        self.per_peak_bg = np.concatenate(([0, ], self.blobsouter[:, ImageD11.cImageD11.mx_I]))
        self.bgcalc = self.per_peak_bg[self.labels]
        self.m_top = self.cor > self.bgcalc * self.bgc
        self.forprops = self.cor * self.m_top
        self.blobs = ImageD11.cImageD11.blobproperties(self.forprops,
                                                       self.labels * self.m_top,
                                                       self.npks,
                                                       omega=omega)
        ImageD11.cImageD11.blob_moments(self.blobs)
        self.enoughpx = self.blobs[:, ImageD11.cImageD11.s_1] >= self.minpx
        self.goodpeaks = self.blobs[self.enoughpx]
        return self.goodpeaks

    def plots(self):
        pass


@functools.lru_cache(maxsize=1)
def get_dset(h5name, dsetname):
    """This avoids to re-read the dataset many times"""
    dset = h5py.File(h5name, "r")[dsetname]
    return dset


def pps(arg):
    hname, dsetname, num, omega, bgfile, split_range, key1_length, key2_length = arg
    if pps.worker is None:
        pps.worker = worker(bgfile)
    if split_range:
        keys, detector = dsetname.split("/", maxsplit=1)
        key1, key2 = keys.split("_")
        key1_dsetname = "/".join((key1, detector))
        key2_dsetname = "/".join((key2, detector))
        if num < key1_length:
            frm = get_dset(hname, key1_dsetname)[num]
        else:
            frm = get_dset(hname, key2_dsetname)[num-key1_length]
    else:
        frm = get_dset(hname, dsetname)[num]
    pks = pps.worker.peaksearch(frm, omega=omega)
    return num, pks


pps.worker = None

PKSAVE = 's_raw f_raw o_raw s_1 s_I'.split()
PKCOL = [getattr(ImageD11.cImageD11, p) for p in PKSAVE]


#    s_1, s_I, s_I2,\
#    s_fI, s_ffI, s_sI, s_ssI, s_sfI, s_oI, s_ooI, s_foI, s_soI, \
#    bb_mn_f, bb_mn_s, bb_mx_f, bb_mx_s, bb_mn_o, bb_mx_o, \
#    mx_I, mx_I_f, mx_I_s, mx_I_o, dety, detz, \
#    avg_i, f_raw, s_raw, o_raw, f_cen, s_cen, \
#    m_ss, m_ff, m_oo, m_sf, m_so, m_fo


def process(hname, scan, detector, omeganame, bgfile, houtname, ncpu):

    # "Scan" here is the key inside the specific h5 file
    # Normally there is only one key like 1.1 or 2.2
    # But for 304m_DLS3 there are two omega ranges
    # Need to check for two keys

    if len(scan.split("_")) != 1:
        # We're working on a split-range scan
        key1, key2 = scan.split("_")
        with h5py.File(hname, 'r') as hin:
            key1_length = hin[key1][detector].shape[0]
            key2_length = hin[key2][detector].shape[0]

            shp = (key1_length + key2_length, hin[key1][detector].shape[1], hin[key1][detector].shape[2])
            key1_omega = hin[key1][omeganame][()] % 360
            key2_omega = hin[key2][omeganame][()] % 360
            log.info(len(key1_omega))
            log.info(len(key2_omega))
            omega = np.concatenate([key1_omega, key2_omega])
        split_range = True

    else:
        with h5py.File(hname, 'r') as hin:
            shp = hin[scan][detector].shape
            omega = hin[scan][omeganame][()] % 360
        split_range = False

        key1_length = 0
        key2_length = 0

    log.info(shp)
    log.info(len(omega))
    assert len(omega) == shp[0]
    detname = "/".join((scan, detector))
    args = [(hname, detname, i, omega[i], bgfile, split_range, key1_length, key2_length) for i in range(shp[0])]
    t0 = timeit.default_timer()

    with h5py.File(houtname, "a") as hout:
        g = hout.require_group(scan + '/peaks2d')
        g.attrs['h5input'] = hname
        g.attrs['h5scan'] = scan
        g.attrs['detector_path'] = detector
        g.attrs['omega_path'] = omeganame
        g.attrs['bgfile'] = bgfile
        g.attrs['script_source'] = open(__file__, 'rb').read()
        g['omega'] = omega
        g['npks'] = np.zeros(len(omega), int)
        for name in PKSAVE:
            g.create_dataset(name,
                             dtype=np.float32,
                             shape=(1,),
                             chunks=(4096,),
                             maxshape=(None,))

        npks = 0
        with multiprocessing.Pool(ncpu) as mypool:
            i = -1
            for num, pks in tqdm.tqdm(mypool.imap(pps, args), total=shp[0]):
                i += 1
                if len(pks) == 0:
                    continue
                npk = len(pks)
                npks += npk
                g['npks'][i] = npk
                for name, col in zip(PKSAVE, PKCOL):
                    g[name].resize((npks,))
                    g[name][-npk:] = pks[:, col]
    # closes, flushes, now merge
    log.info("\nMerging in 3D")
    do3dmerge(houtname, scan)