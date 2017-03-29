#!/usr/bin/python

'''
Simple script to calculated basic statistics on a chain

Date: 2/14/2017
'''

import numpy as np
import h5py
import argparse
import scipy.stats as sci_stats
from scipy import signal
from math import floor

parser = argparse.ArgumentParser(
	description='Simple script to calculated basic ' + 
			'statistics on a chain')
parser.add_argument('chains', type=str,
	help='Chains on which statistics will be calculated.' +
		'Seperate multiple chains with commas (no spaces).')
parser.add_argument('-o', '--output', type=str,
	help='File to store output. If none is provided, ' +
		'results are printed to stdout.')
args = parser.parse_args()

params = ['a_spin1', 'a_spin2', 'chirpmass', 'costheta_jn', 'declination', 'deltalogl', 'logdistance', 'phi12', 'phi_jl', 'polarisation', 'q', 'rightascension', 'tilt_spin1', 'tilt_spin2']

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
		raise ACLError('autocorrelation length too short for consistent estimate')

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
def process(chain_file):
	try:
		raw_data = h5py.File(chain_file)[r'/lalinference/lalinference_mcmc/posterior_samples']
	except KeyError:
		print('[ERROR] Chain file not processed correctly: %s' % chain_file)
		return None

	cycle_index = raw_data.dtype.names.index('cycle')
	data = {}
	indices = {}
	for param in params:
		data[param] = []
		indices[param] = raw_data.dtype.names.index(param)

	trunc_data = []
	for i in range(len(raw_data)):
		if (raw_data[i][cycle_index] >= 0):
			trunc_data = raw_data[i:]
			break
	for i in range(len(trunc_data)):
		for param in params:
			data[param].append(trunc_data[i][indices[param]])

	return data

#Returns dictionary of basic statistcs
def calculate_stats(data, param):
	stats = {}
	stats['mean'] = np.mean(data[param])
	stats['median'] = np.median(data[param])
	stats['std_dev'] = np.std(data[param])
	stats['eff_samp_size'], stats['autocorr'] = effectiveSampleSize(data[param])

	return stats

#helper method that builds the final output
def build_output(stats, param):
	output = 'Statistics for %s:\n' % param
	output += 'MEAN: %f\n' % stats['mean']
	output += 'MEDIAN: %f\n' % stats['median']
	output += 'STD_DEV: %f\n' % stats['std_dev']
	output += 'AUTOCORR: %.2f\n' % stats['autocorr']
	output += 'EFFECTIVE_SAMPLE_SIZE: %d\n' % stats['eff_samp_size']
	
	return output

def write_to_file(content, file_path):
	with open(file_path, 'w') as file:
		file.write(content)

###---(^_^)---MAIN_IMPLEMENTATION---(^_^)---###
output = "RESULTS\n"
output +="------------------------------------------------\n\n"
chains = args.chains.split(',')
for chain in chains:
	print('[INFO] Processing %s' % chain)
	data = process(chain)
	if (data is not None):
		output += 'CHAIN %s:' % chain + '\n'
		acl_max = ('none', 0)
		for param in params:
			print('[INFO] Calculating statistics for %s' % param)
			stats = calculate_stats(data, param)
			if (stats['autocorr'] > acl_max[1]):
				acl_max = (param, stats['autocorr'])
			output += build_output(stats, param) + '\n'
		output += '%s HAS THE  LARGEST ACL:\n' % acl_max[0]
		output += build_output(calculate_stats(data, acl_max[0]), acl_max[0])
	output += '------------------------------------------------\n'
print('[INFO] Done!\n')

if (args.output):
	write_to_file(output, args.output)
else:
	print(output)

