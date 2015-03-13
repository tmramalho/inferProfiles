import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.ndimage import filters
from sklearn.decomposition import PCA
import scipy.stats as stats
import re

import config
reload(config)

def moving_average(series, sigma=3, ks=27):
	b = gaussian(ks, sigma)
	average = filters.convolve1d(series, b / b.sum())
	var = filters.convolve1d(np.power(series - average, 2), b / b.sum())
	return average, var

def ensure_dir(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
		os.makedirs(d)
	return f

def plot_pca(y, pca, yp, L, name):
	plt.figure(figsize=(12,8))
	lx = np.linspace(np.min(L), np.max(L), 100)
	x = np.linspace(0,1,y.shape[1])
	plt.subplot(231)
	plt.scatter(yp[:,0], yp[:,1], c=L/float(np.max(L)), cmap=plt.get_cmap('jet'))
	plt.subplot(232)
	m,b,r,p,s = stats.linregress(L, yp[:,0])
	plt.scatter(L, yp[:,0])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc1 r:{0:.2f},p:{1:.2e}".format(r,p))
	plt.subplot(233)
	m,b,r,p,s = stats.linregress(L, yp[:,1])
	plt.scatter(L, yp[:,1])
	plt.plot(lx, m*lx+b, color='r')
	plt.fill_between(lx, (m-s)*lx+b, (m+s)*lx+b, alpha=0.3, color='r')
	plt.title("pc2 r:{0:.2f},p:{1:.2e}".format(r,p))
	plt.subplot(234)
	plt.plot(x, y.T, alpha=0.5)
	plt.title('data')
	plt.subplot(235)
	n_samples = 50
	s = np.random.normal(scale=np.std(yp[:,0]), size=n_samples)
	v = np.vstack([s, np.zeros(n_samples)]).T
	yt = pca.inverse_transform(v)
	plt.plot(x, yt.T, alpha=0.5)
	plt.title('pc1')
	plt.subplot(236)
	n_samples = 50
	s = np.random.normal(scale=np.std(yp[:,1]), size=n_samples)
	v = np.vstack([np.zeros(n_samples), s]).T
	yt = pca.inverse_transform(v)
	plt.plot(x, yt.T, alpha=0.5)
	plt.title('pc2')
	plt.savefig(ensure_dir(os.path.join(config.plots_path, "full_mutant", 
                                          "pca_{0}.pdf".format(name))))
	plt.clf()

def do_pca_analysis(profiles, lens, name='', plot=False, print_debug=False):
      L = np.array(0.446*(lens-np.mean(lens)), dtype='float64')
      pr = []
      for i,p in enumerate(profiles):
		mask = np.isnan(p)
		p[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), p[~mask])
		av, va = moving_average(np.log(p+0.001), 46, 100)
		pr.append(av)
      y = np.array(pr)
      pca = PCA(n_components=2)
      pca.fit(y)
      if print_debug:
		print pca.explained_variance_ratio_
      yp = pca.transform(y)
	# L = L[np.where(L > -40)]
	# yp = yp[np.where(L > -40)]
      m,b,r,p,_ = stats.linregress(L, yp[:,0])
      p1 = [p]
      r1 = [r]
      for _ in xrange(1000):
		sample = np.random.choice(L.shape[0], L.shape[0], replace=True)
		m,b,r,p,_ = stats.linregress(L[sample], yp[sample,0])
		p1.append(p)
		r1.append(r)
      m,b,r,p,_ = stats.linregress(L, yp[:,1])
      p2 = [p]
      r2 = [r]
      for _ in xrange(1000):
		sample = np.random.choice(L.shape[0], L.shape[0], replace=True)
		m,b,r,p,_ = stats.linregress(L[sample], yp[sample,1])
		p2.append(p)
		r2.append(r)
      if plot:
		plot_pca(y, pca, yp, L, name)
      mean_prof = np.mean(y, axis=0)
      dr = np.max(mean_prof) - np.min(mean_prof)
      more_stats_d = {'norm_sigma_l': np.std(lens) / np.mean(lens)}
      return r1, p1, r2, p2, L.shape[0], name, np.std(L), dr, more_stats_d

def plot_ks_analysis(lower_y, upper_y, pval, name):
	plt.figure(figsize=(5,10))
	plt.subplot(311)
	ax = plt.gca()
	ax.set_axis_bgcolor('white')
	plt.title('smaller embryos, N={0}'.format(lower_y.shape[0]))
	for i,p in enumerate(pval):
		if p < 0.01:
			plt.axvspan(i, i+1, facecolor='0.5', alpha=0.5)
		elif p < 0.05:
			plt.axvspan(i, i+1, facecolor='0.5', alpha=0.2)
	plt.plot(np.exp(lower_y.T[:,:100]))
	plt.subplot(312)
	plt.title('bigger embryos, N={0}'.format(upper_y.shape[0]))
	plt.plot(np.exp(upper_y.T[:,:100]))
	plt.subplot(313)
	plt.plot(pval)
	plt.plot(np.ones(shape=len(pval))*0.05, "g--", label=r"5\% confidence")
	plt.plot(np.ones(shape=len(pval))*0.01, "r--", label=r"1\% confidence")
	plt.yscale('log')
	plt.ylabel('pvalue', fontsize=10)
	plt.xlabel('x/L')
	plt.suptitle(name)
	plt.savefig(ensure_dir(os.path.join(config.plots_path,
                                          "full_mutant", "ks_{0}.pdf".format(name))))
	plt.clf()

def do_ks_analysis(profiles, lens, name='', plot=False):
	L = np.array(0.446*(lens-np.mean(lens)), dtype='float64')
	n, bins = np.histogram(L, bins=2)
	idx_l = np.digitize(L, bins)
	pos_l = (idx_l == 1)
	r_list_l = np.where(pos_l)[0]
	lower_y = np.log(np.array(list(profiles[r_list_l]), dtype=np.float))
	# print lower_y.shape, L[r_list_l]
	pos_u = (idx_l > 1)
	r_list_u = np.where(pos_u)[0]
	upper_y = np.log(np.array(list(profiles[r_list_u]), dtype=np.float))
	# print upper_y.shape, L[r_list_u]
	if upper_y.shape[0] < 2 or lower_y.shape[0] < 2:
		return np.ones(profiles[0].shape[0])
	'''ks 2 sample'''
	pval=[]
	for k in xrange(lower_y.shape[1]):
		try:
			_, p = stats.ks_2samp(lower_y[:,k], upper_y[:,k])
		except ValueError:
			p = 1
		pval.append(p)
	if plot:
		plot_ks_analysis(lower_y, upper_y, pval, name)
	pv = np.array(pval)
	return pv

def trim_name(string):
	string = string[:-7]
	string = re.sub('data_dorsal_\d\d\d\d\d\d_Kni_Kr_Gt_Hb_','',string, flags=re.I)
	string = re.sub('orer','',string, flags=re.I)
	string = string.replace("_","")
	string = string.replace("class1","")
	return string

def summary_plot(fr):
	cc = plt.rcParams['axes.color_cycle']
	c_gap = {'Kni':cc[0], 'Kr ':cc[1], 'Hb ':cc[2], 'Gt ':cc[3]}
	for i,k in enumerate(fr):
		pp = 0
		for r in fr[k]:
			if r[5][-4:] != 'late':
				continue
			plt.plot((i+0.1+pp,i+0.1+pp), (np.percentile(r[1], 5), np.percentile(r[1], 95)), c=c_gap[r[5][:3]], alpha=0.5)
			plt.plot(i+0.1+pp, np.median(r[1]), c=c_gap[r[5][:3]], marker='o', mew=0)
			plt.plot((i-0.1-pp,i-0.1-pp), (np.percentile(r[3], 5), np.percentile(r[3], 95)), c=c_gap[r[5][:3]], alpha=0.5)
			plt.plot(i-0.1-pp, np.median(r[3]), c=c_gap[r[5][:3]], marker='o', mew=0)
			pp += 0.1
	n = len(fr.keys())
	plt.plot(np.linspace(-0.5, n-0.5, 10), np.ones(shape=10)*0.05, "g--", label=r"5\% confidence")
	plt.plot(np.linspace(-0.5, n-0.5, 10), np.ones(shape=10)*0.01, "r--", label=r"1\% confidence")
	rects = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in c_gap.values()]
	leg = plt.legend(rects, c_gap.keys(), loc='lower right')
	leg.get_frame().set_alpha(0.5)
	names = [trim_name(k) for k in fr.keys()]
	plt.xticks(np.arange(n), names, rotation=90, size='small')
	plt.xlim([-0.5, n-0.5])
	plt.yscale('log')
	plt.ylabel('pvalue')
	for xl in np.arange(n-1):
		plt.plot((xl+0.5,xl+0.5), plt.ylim(), 'k-', alpha=0.5)
	plt.savefig(os.path.join(config.plots_path, "full_mutant", "summary_median.pdf"))
	plt.clf()
	for i,k in enumerate(fr):
		for r in fr[k]:
			if r[5][-4:] != 'late':
				continue
			plt.plot(i+0.2, np.mean(r[1]), c=c_gap[r[5][:3]], marker='o', mew=0)
			plt.plot(i-0.2, np.mean(r[3]), c=c_gap[r[5][:3]], marker='o', mew=0)
	n = len(fr.keys())
	plt.plot(np.linspace(-0.5, n-0.5, 10), np.ones(shape=10)*0.05, "g--", label=r"5\% confidence")
	plt.plot(np.linspace(-0.5, n-0.5, 10), np.ones(shape=10)*0.01, "r--", label=r"1\% confidence")
	rects = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in c_gap.values()]
	leg = plt.legend(rects, c_gap.keys(), loc='lower right')
	leg.get_frame().set_alpha(0.5)
	names = [trim_name(k) for k in fr.keys()]
	plt.xticks(np.arange(n), names, rotation=90, size='small')
	plt.xlim([-0.5, n-0.5])
	plt.yscale('log')
	plt.ylabel('pvalue')
	for xl in np.arange(n-1):
		plt.plot((xl+0.5,xl+0.5), plt.ylim(), 'k-', alpha=0.5)
	plt.savefig(os.path.join(config.plots_path, "full_mutant", "summary_mean.pdf"))
	plt.clf()
	l_var = []
	size = []
	for i,k in enumerate(fr):
		l_var.append(fr[k][0][6])
		size.append(fr[k][0][4])
	plt.plot(np.arange(n), l_var)
	plt.ylabel('var L')
	plt.twinx()
	plt.plot(np.arange(n), size, c='r')
	plt.ylabel('n')
	plt.xticks(np.arange(n), names, rotation=90, size='small')
	plt.savefig(os.path.join(config.plots_path, "full_mutant", "sum_diag.pdf"))
	plt.clf()

def rejection_analysis(fr, fkr):
	fr.pop('data_dorsal_130420_Kni_Kr_Gt_Hb_BcdNosTsl_OreR_AP.mat', None)
	rejection_matrix = np.zeros((len(fr.keys()), len(fr[fr.keys()[0]])/2, 2))
	gene_lut = {'Kr':0, 'Kni':1, 'Hb':2, 'Gt':3}
	stage_lut = {'early':0, 'late':1}
	for i,k in enumerate(fr.keys()):
		for n in xrange(len(fr[k])):
			gene_name, mutant_name, stage = fr[k][n][5].split(' ')
			reject_variable = 0
			pv = fkr[k][n]
			reject_ratio = np.where(pv<0.01)[0].shape[0] / float(pv.shape[0])
			p1 = np.mean(fr[k][n][1])
			p2 = np.mean(fr[k][n][3])
			if (p1 < 0.01 or p2 < 0.01) and reject_ratio > 0.1:
				print fr[k][n][5]
				print 'REJECT'
				print p1, p2, reject_ratio
				reject_variable = 1
			rejection_matrix[i, gene_lut[gene_name], stage_lut[stage]] = reject_variable
	plt.subplot(121)
	plt.title('late')
	plt.grid(False)
	plt.imshow(rejection_matrix[:,:,1], interpolation='nearest', cmap='RdYlBu_r')
	plt.xticks(np.arange(len(gene_lut.keys())), sorted(gene_lut, key=gene_lut.get))
	plt.yticks(np.arange(len(fr.keys())), [trim_name(k) for k in fr.keys()])
	plt.subplot(122)
	plt.title('early')
	plt.grid(False)
	plt.imshow(rejection_matrix[:,:,0], interpolation='nearest', cmap='RdYlBu_r')
	plt.xticks(np.arange(len(gene_lut.keys())), sorted(gene_lut, key=gene_lut.get))
	plt.yticks(np.arange(len(fr.keys())), [trim_name(k) for k in fr.keys()])
	plt.savefig(ensure_dir(os.path.join(config.plots_path, 'rejection_matrix.pdf')))

def make_summary_plot(fr, fkr):
	nmax = len(fr.keys())/2+1
	plt.figure(figsize=(6*2, 4*nmax))
	for i,k in enumerate(fr.keys()):
		plt.subplot(nmax, 2, i+1)
		for n in xrange(len(fr[k])):
			dataset = fr[k][n]
			gene_name, mutant_name, stage = dataset[5].split(' ')
			#if stage == 'early':
			#	continue
			p1 = np.mean(dataset[1])
			p2 = np.mean(dataset[3])
			samples = dataset[2*np.argmin([p1,p2])+1]
			pca_std = [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]
			r_samples = np.power(dataset[2*np.argmin([p1,p2])], 2)
			r_sq = np.mean(r_samples)
			r_ci = [[r_sq-np.percentile(r_samples, 2.5)], [np.percentile(r_samples, 97.5)-r_sq]]
			s_size = dataset[4]
			dr = dataset[7]
			col = 'r' if pca_std[1] < 0.05 else 'b'
			plt.errorbar(dr, r_sq, yerr=r_ci,marker='o', ms=np.sqrt(s_size),
	            alpha=0.5, c=col)
			plt.title(trim_name(dataset[5])[3:-1])
			text_an = '{0} {1}'.format(stage, gene_name)
			plt.annotate(text_an, xy = (dr, r_sq), xytext = (-20, 20),
				textcoords = 'offset points', ha = 'right', va = 'bottom',
				bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.2),
				arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', ec='0.1'))
		plt.ylabel('r_sq')
		plt.xlabel('log (fold change)')
	plt.tight_layout()
	plt.savefig(ensure_dir(os.path.join(config.plots_path, 'summary', 
                                          'gap_bubbles.pdf')))
	plt.clf()

def make_paper_plot(fr, fkr):
	plt.figure(figsize=(6*2, 4*2))
	mutants = ['etsl', 'Bcd2X', 'bcdE1']
	gene_index = {'Kr': 1, 'Gt': 2, 'Kni': 3, 'Hb': 4}
	mutant_index = {'etsl': 'o', 'Bcd2X': 'v', 'bcdE1': 's'}
	for i,k in enumerate(fr.keys()):
		mutant_name = trim_name(fr[k][0][5])[4:]
		if mutant_name in mutants:
			for n in xrange(len(fr[k])):
				dataset = fr[k][n]
				gene_name, _, stage = dataset[5].split(' ')
				p1 = np.mean(dataset[1])
				p2 = np.mean(dataset[3])
				samples = dataset[2*np.argmin([p1,p2])+1]
				pca_std = [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]
				r_samples = np.power(dataset[2*np.argmin([p1,p2])], 2)
				r_sq = np.mean(r_samples)
				r_ci = [[r_sq-np.percentile(r_samples, 2.5)], [np.percentile(r_samples, 97.5)-r_sq]]
				s_size = dataset[4]
				dr = dataset[7]
				col = 'r' if pca_std[1] < 0.05 else 'b'
				fill = 'none' if stage == 'early' else None
				gi = gene_index[gene_name]
				plt.subplot(2, 2, gi)
				text_an = '{0} {1} {2}'.format(stage, gene_name, mutant_name)
				plt.errorbar(dr, r_sq, yerr=r_ci,marker=mutant_index[mutant_name], ms=2*np.sqrt(s_size),
					alpha=0.5, c=col, mec=col, mfc=fill, label=text_an)

				# plt.annotate(text_an, xy = (dr, r_sq), xytext = (-20, 20),
				# 	textcoords = 'offset points', ha = 'right', va = 'bottom',
				# 	bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.2),
				# 	arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', ec='0.1'))
	for i in xrange(4):
		plt.subplot(2, 2, i+1)
		leg = plt.legend(loc='upper left', fontsize=9, markerscale=0.5)
		leg.get_frame().set_alpha(0.5)
		plt.ylabel('r_sq')
		plt.xlabel('log (fold change)')
	plt.tight_layout()
	plt.savefig(ensure_dir(os.path.join(config.plots_path, 'summary', 
                                          'paper_gap_bubbles.pdf')))
	plt.clf()

def make_table(fr, fkr):
	for i,k in enumerate(fr.keys()):
		labels = []
		ks_ratio = []
		pca_vals = []
		s_size = []
		test_1 = []
		test_2 = []
		sigma_l = []
		pca_std = []
		r_sq = []
		r_ci = []
		for n in xrange(len(fr[k])):
			dataset = fr[k][n]
			gene_name, mutant_name, stage = dataset[5].split(' ')
			#if stage == 'early':
			#	continue
			labels.append("{0} {1}".format(gene_name, stage))
			pv = fkr[k][n]
			reject_ratio1 = np.where(pv<0.01)[0].shape[0] / float(pv.shape[0])
			p1 = np.mean(dataset[1])
			p2 = np.mean(dataset[3])
			p_pca = np.min([p1, p2])
			pca_vals.append(p_pca)
			samples = dataset[2*np.argmin([p1,p2])+1]
			pca_std.append([np.percentile(samples, 2.5), np.percentile(samples, 97.5)])
			r_samples = np.power(dataset[2*np.argmin([p1,p2])], 2)
			r_sq.append(np.mean(r_samples))
			r_ci.append([np.percentile(r_samples, 2.5), np.percentile(r_samples, 97.5)])
			s_size.append(dataset[4])
			sigma_l.append(dataset[6])
			if p_pca < 0.01 and reject_ratio1 > 0.1:
				test_1.append(1)
			else:
				test_1.append(-1)
			reject_ratio2 = np.where(pv<0.05)[0].shape[0] / float(pv.shape[0])
			if p_pca < 0.05 and reject_ratio2 > 0.1:
				test_2.append(1)
			else:
				test_2.append(-1)
			ks_ratio.append((reject_ratio1, reject_ratio2))

		def saturate_pvals(inp):
			out = np.empty_like(inp)
			out[np.where(inp <= 0.05)] = 1
			out[np.where(inp > 0.05)] = -1
			return 0.5*out

		num_cols = len(labels)
		color_data = np.vstack([
			np.zeros((num_cols,)),
			saturate_pvals(np.array(pca_vals)),
			np.zeros((num_cols,)),
		    np.zeros((num_cols,)),
		    np.zeros((num_cols,)),
			test_1, test_2])
		num_rows = color_data.shape[0]
		plt.figure(figsize=(14, 7))
		ax = plt.gca()
		ax.imshow(color_data, interpolation='nearest', cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

		#text portion
		x_array = np.arange(num_cols)
		y_array = np.arange(num_rows)
		x, y = np.meshgrid(x_array, y_array)
		print num_rows, num_cols

		for x_val, y_val in zip(x.flatten(), y.flatten()):
			if y_val == 0:
				tx = '{0:.02f} (1%)\n{1:.02f} (5%)'.format(ks_ratio[x_val][0], ks_ratio[x_val][1])
				ax.text(x_val, y_val, tx, va='center', ha='center')
			elif y_val == 1:
				tx = '{0:.01e}\n[{1:.02e},{2:.02e}]'.format(pca_vals[x_val], pca_std[x_val][0], pca_std[x_val][1])
				ax.text(x_val, y_val, tx, va='center', ha='center')
			elif y_val == 2:
				tx = '{0}'.format(s_size[x_val])
				ax.text(x_val, y_val, tx, va='center', ha='center')
			elif y_val == 3:
				tx = '{0:.02f}'.format(sigma_l[x_val])
				ax.text(x_val, y_val, tx, va='center', ha='center')
			elif y_val == 4:
				tx = '{0:.02f}\n[{1:.02f},{2:.02f}]'.format(r_sq[x_val], r_ci[x_val][0], r_ci[x_val][1])
				ax.text(x_val, y_val, tx, va='center', ha='center')
			else:
				pass

		ax.set_xticks(x_array+0.5)
		ax.set_yticks(y_array+0.5)
		ax.set_xticklabels(labels, rotation = 90)
		ax.set_yticklabels(['ks', 'pca', 'n_samples', 'sigma_l', 'test1', 'test'])
		plt.title(mutant_name)
		plt.tight_layout()
		plt.savefig(ensure_dir(os.path.join(config.plots_path, 'summary', 
                                               'gap{0}_table.pdf'.format(i))))
		plt.clf()

		ks_ratio = np.array(ks_ratio)
		plt.figure(figsize=(4,10))
		plt.subplot(411)
		plt.bar(np.arange(len(pca_vals)), ks_ratio[:,0] , 0.5)
		plt.bar(np.arange(len(pca_vals))+0.5, ks_ratio[:,1] , 0.5, color='r')
		plt.subplot(412)
		plt.bar(np.arange(len(pca_vals)), pca_vals, 1)
		plt.subplot(413)
		plt.bar(np.arange(len(s_size)), s_size, 1)
		plt.subplot(414)
		plt.bar(np.arange(len(sigma_l)), sigma_l, 1)
		plt.tight_layout()
		plt.savefig(ensure_dir(os.path.join(config.plots_path, 'summary', 
                                               'gap{0}_bar.pdf'.format(i))))
		plt.clf()

if __name__ == '__main__':
	plot_flag = False
	res_path = os.path.join(config.tmp_path, 'mutant_all_res.npy')
	try:
		full_results, full_ks_results = np.load(res_path)
	except IOError:
		print 'no results found. please wait a sec'
		full_results = dict()
		full_ks_results = dict()
		for csv_filename in os.listdir(os.path.join(config.mutant_path)):
			if csv_filename.endswith(".mat"):
				print csv_filename
				results = []
				ks_results = []
				gap_data = sio.loadmat(os.path.join(config.mutant_path, 
                                                           '{0}'.format(csv_filename)),
                                              squeeze_me=True)
				gap_data = gap_data['data']
				pos = (gap_data['age'] >= 40) & (gap_data['age'] <= 50) & (gap_data['orient'] == 1) & (gap_data['genotype'] == 2)
				ind = np.where(pos)[0]
				results.append(do_pca_analysis(gap_data['Kni'][ind], gap_data['AP'][ind], 'Kni {0} late'.format(csv_filename[:-4]), plot=plot_flag))
				results.append(do_pca_analysis(gap_data['Kr'][ind], gap_data['AP'][ind], 'Kr {0} late'.format(csv_filename[:-4]), plot=plot_flag))
				results.append(do_pca_analysis(gap_data['Hb'][ind], gap_data['AP'][ind], 'Hb {0} late'.format(csv_filename[:-4]), plot=plot_flag))
				results.append(do_pca_analysis(gap_data['Gt'][ind], gap_data['AP'][ind], 'Gt {0} late'.format(csv_filename[:-4]), plot=plot_flag))
				ks_results.append(do_ks_analysis(gap_data['Kni'][ind], gap_data['AP'][ind], 'Kni {0} late'.format(csv_filename[:-4]), plot=plot_flag))
				ks_results.append(do_ks_analysis(gap_data['Kr'][ind], gap_data['AP'][ind], 'Kr {0} late'.format(csv_filename[:-4]), plot=plot_flag))
				ks_results.append(do_ks_analysis(gap_data['Hb'][ind], gap_data['AP'][ind], 'Hb {0} late'.format(csv_filename[:-4]), plot=plot_flag))
				ks_results.append(do_ks_analysis(gap_data['Gt'][ind], gap_data['AP'][ind], 'Gt {0} late'.format(csv_filename[:-4]), plot=plot_flag))
				pos = (gap_data['age'] >= 15) & (gap_data['age'] <= 25) & (gap_data['orient'] == 1) & (gap_data['genotype'] == 2)
				ind = np.where(pos)[0]
				results.append(do_pca_analysis(gap_data['Kni'][ind], gap_data['AP'][ind], 'Kni {0} early'.format(csv_filename[:-4]), plot=plot_flag))
				results.append(do_pca_analysis(gap_data['Kr'][ind], gap_data['AP'][ind], 'Kr {0} early'.format(csv_filename[:-4]), plot=plot_flag))
				results.append(do_pca_analysis(gap_data['Hb'][ind], gap_data['AP'][ind], 'Hb {0} early'.format(csv_filename[:-4]), plot=plot_flag))
				results.append(do_pca_analysis(gap_data['Gt'][ind], gap_data['AP'][ind], 'Gt {0} early'.format(csv_filename[:-4]), plot=plot_flag))
				ks_results.append(do_ks_analysis(gap_data['Kni'][ind], gap_data['AP'][ind], 'Kni {0} early'.format(csv_filename[:-4]), plot=plot_flag))
				ks_results.append(do_ks_analysis(gap_data['Kr'][ind], gap_data['AP'][ind], 'Kr {0} early'.format(csv_filename[:-4]), plot=plot_flag))
				ks_results.append(do_ks_analysis(gap_data['Hb'][ind], gap_data['AP'][ind], 'Hb {0} early'.format(csv_filename[:-4]), plot=plot_flag))
				ks_results.append(do_ks_analysis(gap_data['Gt'][ind], gap_data['AP'][ind], 'Gt {0} early'.format(csv_filename[:-4]), plot=plot_flag))
				full_results[csv_filename] = results
				full_ks_results[csv_filename] = ks_results
		np.save(res_path, [full_results, full_ks_results])
	#summary_plot(full_results)
	#rejection_analysis(full_results, full_ks_results)
	#make_table(full_results, full_ks_results)
	#make_summary_plot(full_results, full_ks_results)
	make_paper_plot(full_results, full_ks_results)