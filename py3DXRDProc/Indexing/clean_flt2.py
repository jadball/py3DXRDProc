import numpy as np
from matplotlib import pyplot as plt

import ImageD11.columnfile
import ImageD11.unitcell




def unitcell_peaks_mask(cf, dstol, dsmax, rings=None):
    cell = ImageD11.unitcell.unitcell_from_parameters(cf.parameters)
    cell.makerings(dsmax)
    m = np.zeros(cf.nrows, bool)
    if rings is None:
        rings = list(range(len(cell.ringds)))
    for ringid, v in enumerate(cell.ringds):
        if ringid in rings:
            if v < dsmax:
                m |= (abs(cf.ds - v) < dstol)

    return m


def strongest_peaks(colf, uself=True, frac=0.995, B=0.2, doplot=None):
    # correct intensities for structure factor (decreases with 2theta)
    cor_intensity = colf.sum_intensity * (np.exp(colf.ds * colf.ds * B))
    if uself:
        lf = ImageD11.refinegrains.lf(colf.tth, colf.eta)
        cor_intensity *= lf
    order = np.argsort(cor_intensity)[::-1]  # sort the peaks by intensity
    sortedpks = cor_intensity[order]
    cums = np.cumsum(sortedpks)
    cums /= cums[-1]
    enough = np.searchsorted(cums, frac)
    # Aim is to select the strongest peaks for indexing.
    cutoff = sortedpks[enough]
    mask = cor_intensity > cutoff
    if doplot is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(cums / cums[-1], ',')
        axs[0].set(xlabel='npks', ylabel='fractional intensity')
        axs[0].plot([mask.sum(), ], [frac, ], "o")
        axs[1].plot(cums / cums[-1], ',')
        axs[1].set(xlabel='npks logscale', ylabel='fractional intensity', xscale='log', ylim=(doplot, 1.),
                   xlim=(np.searchsorted(cums, doplot), len(cums)))
        axs[1].plot([mask.sum(), ], [frac, ], "o")
        plt.show()
    return mask


def selectpeaks(cf, dstol=0.005, dsmax=10, frac=0.99, doplot=None, rings=None):
    m = unitcell_peaks_mask(cf, dstol=dstol, dsmax=dsmax, rings=rings)
    cfc = cf.copy()
    cfc.filter(m)
    ms = strongest_peaks(cfc, frac=frac, doplot=doplot)
    cfc.filter(ms)
    return cfc


def clean_flt2(infile, parfile, outfile, frac, dsmax, dstol, rings=None):
    print(f'New filter! Working with {frac} {dsmax} {dstol} {rings}')
    cf = ImageD11.columnfile.columnfile(infile)
    cf.parameters.loadparameters(parfile)
    cf.updateGeometry()
    
    cfc = selectpeaks(cf, dstol=dstol, dsmax=dsmax, frac=frac, rings=rings)
    cfc.writefile(outfile)
    
