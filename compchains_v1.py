#!/usr/bin/python

'''
Revised script to calculated basic statistics on a chain

Date: 3/29/2017
Original by William Tong
'''

import numpy as np
import h5py
import argparse
import scipy.stats as sci_stats
from scipy import signal
from math import floor, ceil

import parse_injection

parser = argparse.ArgumentParser(
    description='Simple script to calculated basic statistics on a chain')
parser.add_argument('chains', nargs='+',
    help='Chains on which statistics will be calculated.')
parser.add_argument('-s', "--sanity-check", action="store_true",
    help='Ensure all parameter values are sane. Note that this will not enforce prior boundaries unless those boundaries are the same as physical boundaries (e.g. chirpmass < 0).')
parser.add_argument('-a', '--all', action="store_true",
    help='Output statistics on all parameters in file, including parameters that are not sampled. These latter parameters are not included in the ACL calculations')
parser.add_argument('-i', '--injection',
    help='If provided, will calculate the distance between the injected value and the predicted value for a sampled parameter.')
parser.add_argument('-e', '--event', type=int,
    help='The event number used in conjunction with the injection file.')
args = parser.parse_args()

if (args.injection and args.event is None):
    print('[ERROR]: Please specify an event number to use with the injection file')
    exit()

params = [] #Yes ACL calculation
no_params = [] #No ACL calculation

#params = ['a_spin1', 'a_spin2', 'chirpmass', 'costheta_jn', 'declination', 'logdistance', 'phi12', 'phi_jl', 'polarisation', 'q', 'rightascension', 'tilt_spin1', 'tilt_spin2'] #, 'cycle', "logprior", "deltalogl"]
#params = ['a1', 'a2', 'mc', 'costheta_jn', 'dec', 'logdistance', 'phi12', 'phi_jl', 'psi', 'q', 'ra', 'tilt1', 'tilt2'] #, 'cycle', "logprior", "deltalogl"]

#Params that shouldn't be included in the ACL calculation
not_sampled_params = ('snr', 'logl', 'logprior', 'logpost', 'cycle', 'time')

#Code borrowed from Chris to load injection files
def load_injections(fname):
    from glue.ligolw import lsctables, utils, ligolw
    lsctables.use_in(ligolw.LIGOLWContentHandler)
    xmldoc = utils.load_filename(fname, contenthandler=ligolw.LIGOLWContentHandler)
    return lsctables.SimInspiralTable.get_table(xmldoc)

if (args.injection is not None):
    sampled_params = load_injections(args.injection).columnnames

# Physical parameter limits
ranges = {
    "m1": (0., float("inf")),
    "m2": (0., float("inf")),
    "q" : (0., 1.),
    "chirpmass": (0., float("inf")),
    "a_spin1": (0., 1.),
    "a_spin2": (0., 1.),
    "tilt_spin1": (0., np.pi),
    "tilt_spin2": (0., np.pi),
    "polarization": (0., np.pi),
    "declination": (-np.pi/2., np.pi/2),
    "rightascension": (0., 2*np.pi),
    "phi12": (0., 2*np.pi),
    "phi_jl": (0., 2*np.pi),
    # NOTE: I acos it, but dont change the column name, so the range is
    # mismatched with the column name.
    "cos_thetajn": (-np.pi, np.pi),
    # NOTE: I exp it, but dont change the column name, so the range is
    # mismatched with the column name.
    "logdistance": (0., float("inf")),
}

#Code borrowed from bayespputils.py
class ACLError(StandardError):
    def __init__(self, *args):
        super(ACLError, self).__init__(*args)

def autocorrelation(series):
    x=series-np.mean(series)
    y=np.conj(x[::-1])

    acf=np.fft.ifftshift(signal.fftconvolve(y,x,mode='full'))

    N=series.shape[0]

    acf = acf[0:N]

    return acf/acf[0]


def autocorrelation_length_estimate(series, acf=None, M=5, K=2):
    if acf is None:
        acf=autocorrelation(series)
    acf[1:] *= 2.0
    imax=int(acf.shape[0]/K)

    cacf=np.cumsum(acf)
    s=np.arange(1, cacf.shape[0]+1)/float(M)

    estimates=np.flatnonzero(cacf[:imax] < s[:imax])

    if estimates.shape[0] > 0:
        return s[estimates[0]]
    else:
        raise ACLError('[ERROR] autocorrelation length too short for consistent estimate')

def effectiveSampleSize(samples, Nskip=1):
    samples = np.array(samples)
    N = len(samples)
    acf = autocorrelation(samples[N/2:])
    try:
        acl = autocorrelation_length_estimate(samples[N/2:], acf=acf)
    except ACLError:
        acl = N
    Neffective = floor(N/acl)
    acl *= Nskip

    return (Neffective, acl)

#returns dictionary of values containing relvant data
def process(chain_file, sanitize=True):
    global params
    global no_params

    if chain_file.endswith(".h5"):
        try:
            #raw_data = h5py.File(chain_file, "r")[r'/lalinference/lalinference_mcmc/posterior_samples'][:]
            raw_data = h5py.File(chain_file, "r")[r'/lalinference/lalinference_mcmc/']
            raw_data = raw_data[raw_data.keys()[0]][:]
        except KeyError:
            print('[ERROR] Chain file not processed correctly: %s' % chain_file)
            return None
    else:
        raw_data = np.genfromtxt(chain_file, skip_header=10, names=True)

    cut = raw_data["cycle"] >= 0
    raw_data = raw_data[cut]
    if len(raw_data) == 0:
        print "[INFO] Not out of burnin."
        exit(0)

    #Separate params into params and no_params
    for raw_param in raw_data.dtype.names:
        skip = False
        for no_param in not_sampled_params:
            if no_param in raw_param:
                no_params.append(raw_param)
                skip = True
                break
        if (not skip):
            params.append(raw_param)
                

    for p in params:
        if "log" == p[:3]:
            print "exp %s" % p
            raw_data[p] = np.exp(raw_data[p])
        elif "cos" == p[:3]:
            print "acos %s" % p
            raw_data[p] = np.arccos(raw_data[p])
        elif "sin" == p[:3]:
            print "asin %s" % p
            raw_data[p] = np.arcsin(raw_data[p])
        raw_data = raw_data[~np.isnan(raw_data[p])]

    if sanitize:
        bad = np.zeros(len(raw_data)).astype(bool)
        for p in params:
            bad |= np.isnan(raw_data[p])
            if p not in ranges:
                continue
            bad |= ranges[p][0] > raw_data[p]
            bad |= raw_data[p] > ranges[p][1]
            bcycles = raw_data["cycle"][np.argwhere(bad)]

        print("Bad cycles: " + ", ".join(map(str, bcycles.flatten())))
        if len(bcycles) > 0:
            last = bcycles[-1] / float(raw_data["cycle"][-1]) * 100
            print("Last bad cycles: %d at %4.f%%" % (bcycles[-1], last))
        raw_data = raw_data[~bad]

    return raw_data

def outlier_analysis(data, param, plots=True):
    stats = calculate_stats(data, param)
    mean, std = stats["mean"], stats["std_dev"]

    # FIXME: I think this condition is reversed
    # This is checked in the downsample phase
    # bppu, line 5695
    dll_max = data["logpost"].max()
    cut = (dll_max - data["logpost"]) < 6.5

    d = data[np.argwhere(cut)]
    _, acl = effectiveSampleSize(d[param])
    if acl < 1.0:
        print("WARNING: acl < 1.0: %f, setting to 1.0" % acl)
        acl = 1
    d = d[::int(acl)]

    if plots:
        try:
            h, x, y = np.histogram2d(d["cycle"].flatten(), d[param].flatten(), bins=(59, 19))
            print("----- sample \"plot\" -----")
            for hrow, yi in zip(h.T, y):
                hrow /= max(1., float(hrow.max())) / 9
                row = "".join([str(int(hi)) if hi else " " for hi in hrow])
                print("%8.4f |" % yi + row)
            print(("-" * 80))

            h, b = np.histogram(d[param], bins=20)
            h = h.astype(float)
            print("----- histogram -----")
            h /= h.max()
            h *= 60
            for hi, bi, b_last, b_next in zip(h, b, np.append(b[0] - (b[1] - b[0]), b[0:-1]), np.append(b[1:], float("inf"))):
                if hi >= 0:
                    hi = int(ceil(hi))
                    if "inj_value" in stats.keys():
                        if b_last < stats['inj_value'] < bi:                        
                            print("%8.4f >" % bi + "=" * hi)
                        else:
                            print("%8.4f |" % bi + "=" * hi)
                    else:
                        print("%8.4f |" % bi + "=" * hi)

            print("")
        except ValueError:
            print("Failed to make chain plot")

    std_dist = np.abs(d[param] - mean) / std
    outlier = np.argmax(std_dist)
    outlier_val = d[param][outlier]
    cycle = d["cycle"][outlier]
    logl = d["deltalogl"][outlier]
    logp = d["logprior"][outlier]
    std_dist = std_dist[outlier]
    print("----- Outlier analysis -----")
    print("Outlier value: %f (index %d, cycle %d)" % (outlier_val, outlier, cycle))
    print("Standardized distance: %f" % std_dist)
    print("delta log L: %f log prior: %f" % (logl, logp))

    sane = True
    if param in ranges:
        bad_before = ranges[param][0] > data[param]
        bad_before |= data[param] > ranges[param][1]
        bad_before = len(np.argwhere(bad_before))
        print("number of samples outside of %s range before cuts: %d" % (param, bad_before))
        bad_after = ranges[param][0] > data[param]
        bad_after |= data[param] > ranges[param][1]
        bad_after = len(np.argwhere(bad_after))
        print("number of samples outside of %s range after cuts: %d" % (param, bad_after))
        sane = bad_after == 0
    return sane

#Helper method that retrieves injection values
def get_inj_value(inj_file, event, param):
    values = load_injections(inj_file)[event]
    inj_parser = parse_injection.InjectionParser(values)
    
    if   (param in 'm1 mass1'):
        return inj_parser.inj_m1()
    elif (param in 'm2 mass2'):
        return inj_parser.inj_m2()
    elif (param in 'chirpmass mc chirpm mchirp'):
        return values.mchirp
    elif (param in 'q'):
        return inj_parser.inj_q()
    elif (param in 'ra rightascension'):
        return inj_parser.inj_longitude()
    elif (param in 'dec declination'):
        return values.latitude
    elif (param in 'psi polarisation polarization'):
        return values.polarization
    else:
        spins = inj_parser.inj_spins()
        if   (param in 'a1 a_spin1 aspin1'):
            return spins['a1']
        elif (param in 'a2 a_spin2 aspin2'):
            return spins['a2']
        elif (param in 'tilt_spin1'):
            return spins['tilt1']
        elif (param in 'tilt_spin2'):
            return spins['tilt2']
        elif (param in 'costheta_jn'):
            return np.cos(spins['theta_jn'])
        elif (param in 'phi_jl'):
            return spins['phi_jl']
        elif (param in 'phi12'):
            return spins['phi12']
    
    return None

#Returns dictionary of basic statistcs
def calculate_stats(data, param, do_acl=True):
    stats = {}
    stats['mean'] = np.mean(data[param])
    stats['median'] = np.median(data[param])
    stats['std_dev'] = np.std(data[param])

    if (do_acl):
        stats['eff_samp_size'], stats['autocorr'] = effectiveSampleSize(data[param])

    if (args.injection is not None):
        stats['inj_value'] = get_inj_value(args.injection, args.event, param)
    return stats

#helper method that builds the final output
def build_output(stats, param):
    print('Statistics for %s:' % param)
    print('MEAN: %f' % stats['mean'])
    print('MEDIAN: %d' % stats['median'])
    print('STD_DEV: %f' % stats['std_dev'])
    
    if ('inj_value' in stats.keys()):
        inj_val = stats['inj_value']
        if(inj_val is not None):
            print('INJECTION VALUE: %f' % inj_val)

    if ('autocorr' in stats.keys()):
        print('AUTOCORR: %.2f' % stats['autocorr'])
        print('EFFECTIVE_SAMPLE_SIZE: %d' % stats['eff_samp_size'])

###---(^_^)---MAIN_IMPLEMENTATION---(^_^)---###
print("RESULTS")
print("------------------------------------------------")

all_sane = True
for chain in args.chains:
    print('[INFO] Processing %s' % chain)
    data = process(chain)

    samp, cycles = len(data), data["cycle"].max()
    print('[INFO] %s has %d cycles, %d samples' % (chain, cycles, samp))

    if data is not None:
        print('CHAIN %s:' % chain)

        acl_max = ('none', 0)
        for param in params:
            print('[INFO] Calculating statistics for %s' % param)
            stats = calculate_stats(data, param)
            if (stats['autocorr'] > acl_max[1]):
                acl_max = (param, stats['autocorr'])
            build_output(stats, param)
            all_sane &= outlier_analysis(data, param)
            print('\n')

        if args.all:
            for param in no_params:
                print('[INFO] Calculating statistics for %s' % param)
                stats = calculate_stats(data, param, do_acl=False)
                build_output(stats, param)
                all_sane &= outlier_analysis(data, param)
                print('\n')

        print('%s HAS THE  LARGEST ACL:' % acl_max[0])
        build_output(calculate_stats(data, acl_max[0]), acl_max[0])

    print('------------------------------------------------')

print('[INFO] Done!')

if not all_sane:
    exit(1)
