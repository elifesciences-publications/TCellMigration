import  matplotlib.pylab as pt
import numpy
import utilities
import umap
import sklearn.cluster as cluster
import hdbscan
import seaborn as sns
import scipy.stats
import matplotlib.gridspec as gridspec

htseq_input = 'data_analysis_06_24_2019/lckgfp_counts.txt'
sort_input = 'data_analysis_06_24_2019/lckgfp_index_sort.txt'

htseq_headers,htseq_indexes,htseq_data = utilities.import_tsv(htseq_input)
sort_headers,sort_indexes,sort_data = utilities.import_tsv(sort_input)

transcript_counts_logscale = numpy.log10( htseq_data.T/numpy.sum(htseq_data,axis=1)*10**6 + 1).T
num_genes = numpy.sum(htseq_data > 1.5, axis=1)

####Cluster with UMAP + hdbscan and compare to index sort data

standard_embedding = umap.UMAP(random_state=42).fit_transform(transcript_counts_logscale)
clustering = hdbscan.HDBSCAN(min_samples=10).fit_predict(standard_embedding)
print(clustering)
gene_translation_dict = utilities.translate_gene_names()
htseq_headers_translated = utilities.translate_gene_list(htseq_headers,gene_translation_dict)

for i in range(max(clustering)+1):
	print(i)
	print(numpy.sum(clustering==i))

Tcell_cluster,count = scipy.stats.mode(clustering)

fig,(ax1,ax2) = pt.subplots(1,2,figsize=(9,4))
ax1.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=2, c=clustering)
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')
ax1.text( -.1,1.06, 'A', fontsize=14, transform=ax1.transAxes, fontname="Arial")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.scatter(utilities.get_gene_expression(sort_headers,sort_data,'FSC-A'),utilities.get_gene_expression(sort_headers,sort_data,'BSC-A'), s=1, c=clustering)
ax2.set_xlabel('FSC-A')
ax2.set_ylabel('BSC-A')
ax2.set_xlim(2*10**4,10**6)
ax2.set_ylim(10**4,5*10**6)
ax2.text( -.1,1.06, 'B', fontsize=14, transform=ax2.transAxes, fontname="Arial")
pt.savefig('data_analysis_06_24_2019/umap_hdbscan_sort_09122019.pdf',bbox_inches='tight')


###Classify cells that were not put in a cluster as T cells if they have BSC < 8x10^4

clustering_major = []
classification = []
for i in range(len(clustering)):
	if clustering[i] == -1 and utilities.get_gene_expression(sort_headers,sort_data,'BSC-A')[i] < 8*10**4:
		clustering_major.append(Tcell_cluster)
		classification.append('T cell')
	elif clustering[i] == Tcell_cluster:
		clustering_major.append(Tcell_cluster)
		classification.append('T cell')
	else:
		clustering_major.append(clustering[i])
		classification.append('epithelial cell')

clustering_major = numpy.array(clustering_major)

Tcells = numpy.where(clustering_major == Tcell_cluster)[0]
nonTcells = numpy.where(clustering_major != Tcell_cluster)[0]

output1 = open('data_analysis_06_24_2019/cell_type_classification.txt','w')
for i in range(len(htseq_indexes)):
	output1.write(htseq_indexes[i] + '\t' + classification[i] + '\n')
output1.close()

####Plot the UMAP with T cells and non-T cells labeled

fig = pt.figure(figsize=(5,4))
ax = pt.gca()
ax.scatter(standard_embedding[Tcells, 0], standard_embedding[Tcells, 1], s=2, c='darkblue')
ax.scatter(standard_embedding[nonTcells, 0], standard_embedding[nonTcells, 1], s=2, c='grey')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.text(0,-2,'T cells',color='darkblue',weight='bold')
ax.text(5,-2,'Epithelial',color='grey',weight='bold')
pt.savefig('data_analysis_06_24_2019/umap_hdbscan_classification.pdf',bbox_inches='tight')
pt.close()
#####Perform differential expression analysis on the large cluster relative to the others, and on the sides of the T cell UMAP space
####
diffexp_pvals = []
diffexp_ratios = []
for i in range(htseq_data.shape[1]):
	s,p = scipy.stats.ranksums(htseq_data[Tcells,i],htseq_data[nonTcells,i])
	diffexp_ratio =  numpy.mean(transcript_counts_logscale[Tcells,i]) - numpy.mean(transcript_counts_logscale[nonTcells,i])
	diffexp_pvals.append(p)
	diffexp_ratios.append(diffexp_ratio)
diffexp_ratios = numpy.array(diffexp_ratios)	
diffexp_pvals = numpy.array(diffexp_pvals)
n = 50
diffexp_inds = numpy.argpartition(diffexp_pvals, n)[:n]
diffexp_genes = [htseq_headers_translated[i] for i in diffexp_inds]

for i in diffexp_inds:
	print(htseq_headers_translated[i], diffexp_ratios[i])

diffexp_inds2 = numpy.argpartition(diffexp_pvals,100)[:100]
diffexp_genes2 = numpy.array([htseq_headers_translated[i] for i in diffexp_inds2])
diffexp_pvals2 = diffexp_pvals[diffexp_inds2]
diffexp_ratios2 = diffexp_ratios[diffexp_inds2]
systematic_genes2 = numpy.array([htseq_headers[i] for i in diffexp_inds2])

pval_sort_order = numpy.argsort(diffexp_pvals2)
pvals_sorted = diffexp_pvals2[pval_sort_order]
genes_sorted = diffexp_genes2[pval_sort_order]
ratios_sorted = diffexp_ratios2[pval_sort_order]
genes_syst_sorted = systematic_genes2[pval_sort_order]

diffexp_file = open('data_analysis_06_24_2019/differential_expression.txt','w')
diffexp_file.write(('\t').join(('Gene','Wilcoxon rank-sum p-value','Log10 expression difference (T cells vs. epithelial)')) + '\n')
for i in range(len(genes_sorted)):
	if 'ensembl' in genes_sorted[i]:
		diffexp_file.write(genes_syst_sorted[i] + '\t' + str(pvals_sorted[i]) + '\t' + str(ratios_sorted[i]) + '\n')
	else:
		diffexp_file.write(genes_sorted[i] + '\t' + str(pvals_sorted[i]) + '\t' + str(ratios_sorted[i]) + '\n')
diffexp_file.close()
###

trac_versions = ['ENSDARG00000075807','ENSDARG00000104132']
keratinocyte_markers=['krt8', 'KRT1','aldob', 'ahnak']
Tcell_general = ['tagapa','trac','ptprc','ccr9a','tnfrsf9b','il2rb']
actin_remodeling = ['arpc1b','wasb','arhgdig','coro1a','sept9b','capgb']
Tcell_subcluster = ['ccl38a.5','ccl38.6','zbtb32','scinlb']
nonT_immune = ['igic1s1','ccl35.2','mpx','mpeg1.1','mpeg1.2']

marker_sets = [Tcell_general,actin_remodeling,Tcell_subcluster,keratinocyte_markers,nonT_immune]
marker_labels = ['T cell/immune','ROCK/Rho/Arp2/3','T cell subset','Keratinocyte','Non-T immune']

#####Make a heatmap of the T cells, ordered by UMAP1 coordinate

marker_list_all = []
marker_list_all.extend(Tcell_general)

marker_list_all.extend(actin_remodeling)
#marker_list_all.extend(Tcell_subcluster)
marker_list_all.extend(nonT_immune)

borders = [0,6,12,4,5]

umap_order = numpy.argsort(standard_embedding[Tcells,0])
exp_sorted = transcript_counts_logscale[Tcells,:][umap_order,:]
exp_table = []
for i in range(len(marker_list_all)):
	
	gene_name = marker_list_all[i]
	print(gene_name)
	if gene_name == 'trac':
		e1 = utilities.get_gene_expression(htseq_headers,exp_sorted,trac_versions[0])
		e2 = utilities.get_gene_expression(htseq_headers,exp_sorted,trac_versions[1])
		elog = numpy.log10((e1+e2)/numpy.sum(exp_sorted,axis=1)*10**6 + 1)
		exp_table.append(elog)
	else:
		exp_table.append(utilities.get_gene_expression(htseq_headers_translated,exp_sorted,gene_name))
exp_table = numpy.array(exp_table)
####
pt.figure()
sns.heatmap(exp_table,xticklabels=[],yticklabels=marker_list_all,cbar=True,cbar_kws={'label': 'Log10 CPM'})
ax = pt.gca()

ax.axhline(6,0,1,color='k')
ax.axhline(12,0,1,color='k')
ax.text(-54,3,'T cell/immune',fontsize=8,weight='bold',verticalalignment='center',rotation=90)
ax.text(-54,9,'WASP/ARP2/3',fontsize=8,weight='bold',verticalalignment='center',rotation=90)
ax.text(-54,14.5,'Non-T cell',fontsize=8,weight='bold',verticalalignment='center',rotation=90)
ax.xaxis.set_ticks_position('top')
pt.tick_params(length=0)
ax.set_yticklabels(marker_list_all,fontdict={'fontsize':7},fontstyle='italic')


ax.text(170,-.6,'T cells',fontsize=12,weight='bold',horizontalalignment='center')
pt.savefig('data_analysis_06_24_2019/marker_heatmap.pdf',bbox_inches='tight')

####Make a large marker UMAP plot





all_exp_vals = []
for i in range(len(Tcell_general)):
	gene_name = Tcell_general[i]
	
	if gene_name == 'trac':
		e1 = utilities.get_gene_expression(htseq_headers,htseq_data,trac_versions[0])
		e2 = utilities.get_gene_expression(htseq_headers,htseq_data,trac_versions[1])
		elog = numpy.log10((e1+e2)/(2*numpy.sum(htseq_data,axis=1)*10**6) + 1)
		all_exp_vals.extend(elog)
	else:
		all_exp_vals.extend(utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_name))
for j in range(len(actin_remodeling)):
	gene_name = actin_remodeling[i]
	
	if gene_name == 'trac':
		e1 = utilities.get_gene_expression(htseq_headers,htseq_data,trac_versions[0])
		e2 = utilities.get_gene_expression(htseq_headers,htseq_data,trac_versions[1])
		elog = numpy.log10((e1+e2)/numpy.sum(htseq_data,axis=1)*10**6 + 1)
		all_exp_vals.extend(elog)
	else:
		all_exp_vals.extend(utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_name))
###
max_exp = numpy.percentile(all_exp_vals,99)
min_exp = numpy.min(all_exp_vals)
fig = pt.figure(figsize=(16,12))
gs = gridspec.GridSpec(5,6)
ax_list = []
for i in range(5):
	markers = marker_sets[i]
	label = marker_labels[i]
	
	for j in range(len(markers)):
	
		ax = pt.subplot(gs[i:i+1,j:j+1])
		gene_name = markers[j]
		ax.set_title(gene_name)
		
		if gene_name == 'trac':
			e1 = utilities.get_gene_expression(htseq_headers,transcript_counts_logscale,trac_versions[0])
			e2 = utilities.get_gene_expression(htseq_headers,transcript_counts_logscale,trac_versions[1])
			elog = numpy.log10((e1+e2)/(2*numpy.sum(transcript_counts_logscale,axis=1))*10**6 + 1)
			ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, vmin=min_exp, vmax = max_exp, c=elog)
			
		else:
			sct = ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, vmin=min_exp, vmax = max_exp, c=utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_name))
	
		if j == 0:
			ax.set_ylabel('UMAP2')
			ax.text(-3,-1.6,label,fontsize=14,weight='bold',rotation=90,verticalalignment='center')
		if i == 4:
			ax.set_xlabel('UMAP1')
#ax = gs[2,5]
#pt.colorbar(sct,cax=ax,label='Log10 CPM')
rect_loc = [5/7,.24,.03,.34]
ax = fig.add_axes(rect_loc)
pt.colorbar(sct,cax=ax,label='Log10 CPM')
pt.tight_layout()
pt.savefig('data_analysis_06_24_2019/SI_UMAP.pdf',bbox_inches='tight')

# fig, axes = pt.subplots(2,6,figsize=(15,5),sharex='row',sharey='col')
# for i in range(len(Tcell_general)):
	# gene_name = Tcell_general[i]
	# ax = axes[0][i]
	# if gene_name == 'trac':
		# e1 = utilities.get_gene_expression(htseq_headers,htseq_data,trac_versions[0])
		# e2 = utilities.get_gene_expression(htseq_headers,htseq_data,trac_versions[1])
		# elog = numpy.log10((e1+e2)/numpy.sum(htseq_data,axis=1)*10**6 + 1)
		# ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, vmin=min_exp, vmax = max_exp, c=elog)
	# else:
		# sct = ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, vmin=min_exp, vmax = max_exp, c=utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_name))
	# #ax.set_xlabel('UMAP 1')
	# if i == 0:
		# ax.set_ylabel('UMAP 2')
	# ax.set_title(gene_name)
	# if i==0:
		# ax.text(-4,-1.6,'T cell/immune',rotation=90,verticalalignment='center',fontsize=14, weight = 'bold')
	# if i == 5:
		# pt.colorbar(sct)
# for i in range(len(actin_remodeling)):
	# gene_name = actin_remodeling[i]
	# ax = axes[1][i]
	
	# ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, vmin=min_exp, vmax = max_exp, c=utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_name))
	# ax.set_xlabel('UMAP 1')
	# if i == 0:
		# ax.set_ylabel('UMAP 2')
	# ax.set_title(gene_name)
	# if i==0:
		# ax.text(-4,-1.6,'ROCK/Rho/Arp2/3',rotation=90,verticalalignment='center',fontsize=14, weight = 'bold')
		
# #pt.colorbar()
# pt.tight_layout()
# pt.savefig('data_analysis_06_24_2019/Fig3_UMAP.pdf',bbox_inches='tight')

###Extended figure


# marker_gene_ids = ['arpc1b', 'ccr9a','tagapa','wasb','ENSDARG00000075807','lck','krt8', 'KRT1','aldob', 'ahnak','mpeg1.1','mpx']
# marker_gene_names =  ['arpc1b', 'ccr9a','tagapa','wasb','trac','lck','krt8', 'KRT1','aldob', 'ahnak','mpeg1.1']
# Tcell_markers = ['ENSDARG00000002353','ENSDARG00000043475','ENSDARG00000055186','ENSDARG00000090728','ENSDARG00000075720','ENSDARG00000075807','ENSDARG00000102525','ENSDARG00000071437']
# Tcell_marker_names = ['tagapa','tagapb','ccr9a','tnfrsf9b','il2rb','trac','lck','ptprc']#,'ENSDARG00000075807']
# actin_remodeling = ['arpc1b','wasb','arhgdig','coro1a','sept9b','capgb','scinlb']
# housekeeping = ['actb1','actb2']
# nonTcell_immune_id = ['ENSDARG00000093272','ENSDARG00000070378','ENSDARG00000019521','ENSDARG00000055290','ENSDARG00000043093','ENSDARG00000041835','ENSDARG00000041923','ENSDARG00000073726','ENSDARG00000058673']
# nonTcell_immune_names = ['igic1s1','ccl35.2','mpx','mpeg1.1','mpeg1.2','ccl38a.5','ccl38.6','zbtb32','nkl.2']

# pt.figure()
# pt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=1, c=num_genes)
# pt.savefig('data_analysis_06_24_2019/umap_numgenes_mitofilter2.pdf',bbox_inches='tight')

# pt.close('all')

# fig, axes = pt.subplots(2,6,figsize=(2*len(marker_gene_ids),8))
# for i in range(len(marker_gene_ids)):
	# i1 = int(numpy.floor(i/6))
	# i2 = i%6
	# ax = axes[i1][i2]
	# gene_id = marker_gene_ids[i]
	# print(gene_id)
	# if 'ENSDARG' not in gene_id:
	
		# ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, c=utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_id))
		
		# ax.set_title(gene_id)
	# else:
		# ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, c=utilities.get_gene_expression(htseq_headers,transcript_counts_logscale,gene_id))
		# ax.set_title(marker_gene_names[i])
# pt.savefig('data_analysis_06_24_2019/marker_genes_mitofilter2.pdf',bbox_inches='tight')

# fig, axes = pt.subplots(2,6,figsize=(2*len(marker_gene_ids),8))
# for i in range(len(marker_gene_ids)):
	# i1 = int(numpy.floor(i/6))
	# i2 = i%6
	# ax = axes[i1][i2]
	# gene_id = marker_gene_ids[i]
	# ax.set_xscale('log')
	# ax.set_yscale('log')
	# ax.set_xlim(10**4,5*10**6)
	# ax.set_ylim(10**2,10**3)
	# print(gene_id)
	# if 'ENSDARG' not in gene_id:
	
		# ax.scatter(utilities.get_gene_expression(sort_headers,sort_data,'FSC-A'), utilities.get_gene_expression(sort_headers,sort_data,'FSC-W'), s=4, c=utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_id))
		
		# ax.set_title(gene_id)
	# else:
		# ax.scatter(utilities.get_gene_expression(sort_headers,sort_data,'FSC-A'), utilities.get_gene_expression(sort_headers,sort_data,'FSC-W'), s=4, c=utilities.get_gene_expression(htseq_headers,transcript_counts_logscale,gene_id))
		# ax.set_title(marker_gene_names[i])
	# ax.set_xlabel('FSC-A')
	# ax.set_ylabel('FSC-W')
# pt.savefig('data_analysis_06_24_2019/marker_genes_fsc_mitofilter2.pdf',bbox_inches='tight')

# fig, axes = pt.subplots(2,6,figsize=(2*len(marker_gene_ids),8))
# for i in range(len(marker_gene_ids)):
	# i1 = int(numpy.floor(i/6))
	# i2 = i%6
	# ax = axes[i1][i2]
	# gene_id = marker_gene_ids[i]
	# ax.set_xscale('log')
	# ax.set_yscale('log')
	# ax.set_xlim(10**4,5*10**6)
	# ax.set_ylim(10**2,10**3)
	# print(gene_id)
	# if 'ENSDARG' not in gene_id:
	
		# ax.scatter(utilities.get_gene_expression(sort_headers,sort_data,'BSC-A'), utilities.get_gene_expression(sort_headers,sort_data,'BSC-W'), s=4, c=utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_id))
		
		# ax.set_title(gene_id)
	# else:
		# ax.scatter(utilities.get_gene_expression(sort_headers,sort_data,'BSC-A'), utilities.get_gene_expression(sort_headers,sort_data,'BSC-W'), s=4, c=utilities.get_gene_expression(htseq_headers,transcript_counts_logscale,gene_id))
		# ax.set_title(marker_gene_names[i])
	# ax.set_xlabel('BSC-A')
	# ax.set_ylabel('BSC-W')
# pt.savefig('data_analysis_06_24_2019/marker_genes_bsc_mitofilter2.pdf',bbox_inches='tight')

# fig, axes = pt.subplots(2,5,figsize=(2*len(Tcell_markers),8))
# for i in range(len(Tcell_markers)):
	# i1 = int(numpy.floor(i/5))
	# i2 = i%5
	# ax = axes[i1][i2]
	# gene_id = Tcell_markers[i]
	# print(gene_id)
	# ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, c=utilities.get_gene_expression(htseq_headers,transcript_counts_logscale,gene_id))
	# ax.set_title(Tcell_marker_names[i])
# pt.savefig('data_analysis_06_24_2019/Tcell_marker_genes_mitofilter2.pdf',bbox_inches='tight')

# fig = pt.figure(figsize=(4,4))
# #fig, axes = pt.subplots(1,3,figsize=(12,4))
# trac0 = trac_versions[0]

# for i in range(len(trac_versions)-1):
	
	# gene_id = trac_versions[i+1]
	# ax = pt.gca()
	# #ax = axes[i]
	# print(gene_id)
	# ax.set_xscale('log')
	# ax.set_yscale('log')
	# ax.scatter(utilities.get_gene_expression(htseq_headers,htseq_data,trac0)+1, utilities.get_gene_expression(htseq_headers,htseq_data,gene_id)+1, s=4)
	# #ax.set_title(trac0 + ' vs ' + gene_id)
	# ax.set_xlabel(trac0)
	# ax.set_ylabel(gene_id)
# pt.savefig('data_analysis_06_24_2019/TRAC_versions.pdf',bbox_inches='tight')

# fig, axes = pt.subplots(2,4,figsize=(2*len(actin_remodeling),8))
# for i in range(len(actin_remodeling)):
	# i1 = int(numpy.floor(i/4))
	# i2 = i%4
	# ax = axes[i1][i2]
	# gene_id = actin_remodeling[i]
	# print(gene_id)
	# ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, c=utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_id))
	# ax.set_title(gene_id)
# pt.savefig('data_analysis_06_24_2019/actin_remodeling_genes_mitofilter2.pdf',bbox_inches='tight')

# fig, axes = pt.subplots(2,4,figsize=(2*len(actin_remodeling),8))
# for i in range(len(actin_remodeling)):
	# i1 = int(numpy.floor(i/4))
	# i2 = i%4
	# ax = axes[i1][i2]
	# gene_id = actin_remodeling[i]
	# print(gene_id)
	# ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, c=utilities.get_gene_expression(htseq_headers_translated,transcript_counts_logscale,gene_id))
	# ax.set_title(gene_id)
# pt.savefig('data_analysis_06_24_2019/actin_remodeling_genes_mitofilter2.pdf',bbox_inches='tight')

# fig, axes = pt.subplots(2,5,figsize=(2*len(nonTcell_immune_id),8))
# for i in range(len(nonTcell_immune_id)):
	# i1 = int(numpy.floor(i/5))
	# i2 = i%5
	# ax = axes[i1][i2]
	# gene_id = nonTcell_immune_id[i]
	# print(gene_id)
	# ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=4, c=utilities.get_gene_expression(htseq_headers,transcript_counts_logscale,gene_id))
	# ax.set_title(nonTcell_immune_names[i])
# pt.savefig('data_analysis_06_24_2019/nonTcell_immune_mitofilter2.pdf',bbox_inches='tight')

# #####
# pt.figure()
# ax = pt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')
# pt.scatter(utilities.get_gene_expression(sort_headers,sort_data,'FSC-A'),utilities.get_gene_expression(sort_headers,sort_data,'BSC-A'), s=1, c=clustering)
# pt.xlabel('FSC-A')
# pt.ylabel('BSC-A')
# pt.savefig('data_analysis_06_24_2019/FSC_SSC_by_cell_cluster_mitofilter2.pdf',bbox_inches='tight')

# pt.figure()
# ax = pt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')
# pt.scatter(utilities.get_gene_expression(sort_headers,sort_data,'FSC-A'),utilities.get_gene_expression(sort_headers,sort_data,'lck: EGFP-A-Compensated'), s=1, c=clustering)
# pt.xlabel('FSC-A')
# pt.ylabel('eGFP')
# pt.savefig('data_analysis_06_24_2019/FSC_GFP_by_cell_cluster_mitofilter2.pdf',bbox_inches='tight')

# pt.figure()
# ax = pt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')
# pt.scatter(utilities.get_gene_expression(sort_headers,sort_data,'FSC-A'),utilities.get_gene_expression(sort_headers,sort_data,'lck: EGFP-A-Compensated'), s=1, c=clustering)
# pt.xlabel('FSC-A')
# pt.ylabel('eGFP')
# pt.savefig('data_analysis_06_24_2019/FSC_GFP_by_cell_cluster_mitofilter2.pdf',bbox_inches='tight')

# #####Find differentially expressed genes

# #diffexp_pvals = []
# #for i in range(transcript_counts_logscale.shape[1]):
# #	s,p = scipy.stats.ranksums(cluster1[:,i],cluster2[:,i])
# #	diffexp_pvals.append(p)

# #diffexp_pvals = numpy.array(diffexp_pvals)
# #n = 50
# #diffexp_inds = numpy.argpartition(diffexp_pvals, n)[:n]

# #diffexp_genes = [transcript_list_translated[i] for i in diffexp_inds]
