import  matplotlib.pylab as pt
import numpy
import utilities
import umap
import sklearn.cluster as cluster
from sklearn.linear_model import LinearRegression
import hdbscan
import seaborn as sns
import scipy.stats
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from samalg import SAM
import samalg.utilities as ut
import pandas as pd
from matplotlib.colors import ListedColormap

###Color scheme setup

pal1 = sns.color_palette()
pal = pal1.as_hex()
cdict = {0:pal[0],1:pal[1],2:pal[2],3:pal[3]}

colors = ["dusty purple","faded green"]
#sns.palplot(sns.xkcd_palette(colors))
pal2 = sns.xkcd_palette([colors[0],colors[1]])
#pal2 = sns.color_palette('xkcd',n_colors=2)
cdict2 = {0:pal2[0],1:pal2[1]}

###Data files

htseq_input = 'data_analysis_03_20_2020/lckgfp_counts.txt'
classification_output = 'data_analysis_03_20_2020/cell_type_classification.txt'
sort_input = 'data_analysis_03_20_2020/lckgfp_index_sort.txt'
sample_list = 'singleTcell_RNA_seq/sample_filename_key.txt'
diffexp_filename = 'data_analysis_03_20_2020/differential_expression.txt'
diffexp_filename_tcells = 'data_analysis_03_20_2020/differential_expression_tcells.txt'

htseq_headers,htseq_indexes,htseq_data = utilities.import_tsv(htseq_input)
sort_headers,sort_indexes,sort_data = utilities.import_tsv(sort_input)
transcript_counts_logscale = numpy.log10( htseq_data.T/numpy.sum(htseq_data,axis=1)*10**6 + 1).T

gene_translation_dict = utilities.translate_gene_names()
htseq_headers_translated = utilities.translate_gene_list(htseq_headers,gene_translation_dict)

####Processing: Use the SAM algorithm to project cells into 2D. Compare this clustering to the index sort data

thresh = .1

#sam=SAM(counts=(htseq_data,htseq_headers,htseq_indexes))
#sam.preprocess_data(sum_norm='cell_median',thresh=thresh) # log transforms and filters the data
#sam.run(k=10) #run with default parameters

#sam_cluster = hdbscan.HDBSCAN(min_samples=10).fit_predict(sam.adata.obsm['X_umap'])

thresh = .1
exp_genes = numpy.sum(transcript_counts_logscale > 10**(-4),axis=0)/transcript_counts_logscale.shape[0] > thresh
standard_embedding = umap.UMAP(random_state=42).fit_transform(transcript_counts_logscale[:,exp_genes])
sam_cluster = hdbscan.HDBSCAN(min_samples=10).fit_predict(standard_embedding)

#fig,(ax1,ax2) = pt.subplots(1,2,figsize=(9,4))

fig = pt.figure(figsize=(9,6.4))
gs = fig.add_gridspec(5,2,wspace=.3,hspace=1.2)

ax1 = fig.add_subplot(gs[0:3,0])
ax2 = fig.add_subplot(gs[0:3,1])
ax3 = fig.add_subplot(gs[3:,:])

ax1.scatter(standard_embedding[:,0],standard_embedding[:,1], s=3, c=[cdict2[s] for s in sam_cluster])
#ax1.scatter(sam.adata.obsm['X_umap'][:,0],sam.adata.obsm['X_umap'][:,1], s=3, c=sam_cluster)

ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')
ax1.text( -.1,1.06, 'A', fontsize=14, transform=ax1.transAxes, fontname="Arial")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.scatter(utilities.get_gene_expression(sort_headers,sort_data,'FSC-A'),utilities.get_gene_expression(sort_headers,sort_data,'BSC-A'), s=2, c=[cdict2[s] for s in sam_cluster])
ax2.set_xlabel('FSC-A')
ax2.set_ylabel('BSC-A')
ax2.set_xlim(2*10**4,10**6)
ax2.set_ylim(10**4,5*10**6)
ax2.text( -.1,1.06, 'B', fontsize=14, transform=ax2.transAxes, fontname="Arial")
#pt.savefig('data_analysis_03_20_2020/SAM_clusters_sort.pdf',bbox_inches='tight')

bsc_sort_vals = utilities.get_gene_expression(sort_headers,sort_data,'BSC-A')

####Classify clusters based on expression of ptprc.
clusters = set(sam_cluster)
tcell_clusters = []
nont_clusters = []

for c in clusters:
    cluster_inds = numpy.where(sam_cluster == c)[0]
    ptprc_exp = utilities.get_gene_expression(htseq_headers,htseq_data,'ENSDARG00000071437')[cluster_inds]
    ptprc_prop = numpy.sum(ptprc_exp > .5)/len(ptprc_exp)
    print(ptprc_prop)
    if ptprc_prop > .5:
        tcell_clusters.append(c)
    else:
        nont_clusters.append(c)

tcell_indexes = []
nont_indexes = []

for i in range(htseq_data.shape[0]):
    cl = sam_cluster[i]
    if cl in tcell_clusters and bsc_sort_vals[i] < 2*10**5:
        tcell_indexes.append(i)
    else:
        nont_indexes.append(i)

Tcells = numpy.array(tcell_indexes)
nonTcells = numpy.array(nont_indexes)

print('Number of T cells: ',len(Tcells))
print('Number of other (epithelial) cells: ', len(nonTcells))


####Record the classifications with the cell indexes

file = open(classification_output,'w')
file.write('Index' + '\t' + 'Classification' + '\n')
for i in range(htseq_data.shape[0]):
    cell_name = htseq_indexes[i]
    if i in Tcells:
        classification = 'T cell'
    else:
        classification = 'epithelial cell'
    file.write(cell_name + '\t' + classification + '\n')
file.close()

###Make violin plots to show some of the T cell and epithelial markers; check markers known to be associated with other T cells

gene_common_names = ['ptprc','trac','tagapa','lcp2a','srgn','arpc1b','coro1a','wasb','capgb','scinlb','krt8','cd81a','ctsla','ahnak','igic1s1','ccl35.2','mpx','mpeg1.1','mpeg1.2']

gene_list = ['ENSDARG00000071437','ENSDARG00000075807','ENSDARG00000002353','ENSDARG00000055955','ENSDARG00000077069','ENSDARG00000027063','ENSDARG00000054610','ENSDARG00000026350','ENSDARG00000099672','ENSDARG00000058348','ENSDARG00000058358','ENSDARG00000036080','ENSDARG00000007836','ENSDARG00000061764','ENSDARG00000093272','ENSDARG00000070378','ENSDARG00000019521','ENSDARG00000055290','ENSDARG00000043093']

htseq_data_norm = (htseq_data.T/numpy.sum(htseq_data,axis=1)).T*10**6 ####Counts per million

###Construct a dataframe with expression vectors for the chosen genes, as well as the cluster assignment categorical variable
htseq_data_norm_log = numpy.log10(htseq_data_norm + 1)
exp_list = []
for i in range(len(gene_list)):
    gene = gene_list[i]
    exp = utilities.get_gene_expression(htseq_headers,htseq_data_norm_log,gene)
    exp_list.append(exp)

exp_list = numpy.array(exp_list).T

    
df1 = pd.DataFrame(exp_list,columns=["GN"+str(g) for g in range(len(gene_common_names))],index=htseq_indexes)
df1["id"] = df1.index
df1["cluster_assignment"] = sam_cluster
df1_long = pd.wide_to_long(df1,["GN"],i="id",j="gene")

df1_long["gene_exp"] = df1_long.index.get_level_values("gene")

sns.violinplot(x="gene_exp",y="GN",hue="cluster_assignment",data=df1_long,ax=ax3,order=[0,1,2,3,4,'',5,6,7,8,9,'',10,11,12,13,'',14,15,16,17,18],cut=0,linewidth=.5,scale='width',inner='point',palette=pal2)
ax3.set_xticks([0,1,2,3,4,6,7,8,9,10,12,13,14,15,17,18,19,20,21])
ax3.set_xticklabels(gene_common_names,fontsize=10,rotation=90)
#ax.legend(['T cell','Endothelial'])

ax3.text(2,5,'Immune/T cell',horizontalalignment='center',verticalalignment='top',fontsize=10)
ax3.text(8,5,'Motliity',horizontalalignment='center',verticalalignment='top',fontsize=10)
ax3.text(13.5,5,'Epithelial',horizontalalignment='center',verticalalignment='top',fontsize=10)
ax3.text(19,5,'Other immune\ncell types',horizontalalignment='center',verticalalignment='top',fontsize=10)

ax3.set_ylim(-.5,5.2)
ax3.set_yticks([0,1,2,3,4])
ax3.set_yticklabels([r'$10^0$',r'$10^1$',r'$10^2$',r'$10^3$',r'$10^4$'])
ax3.set_xticklabels(gene_common_names,fontsize=9,rotation=90)
ax3.set_xlabel('')
ax3.get_legend().remove()
ax3.set_ylabel(r'CPM+1')
ax3.text( -.04,1.06, 'C', fontsize=14, transform=ax3.transAxes, fontname="Arial")
pt.savefig('data_analysis_03_20_2020/Tcells_others_sort_marker_genes.pdf',bbox_inches='tight')

####Look at differential expression between T and non-T cells

diffexp_pvals = []
diffexp_ratios = []
for i in range(htseq_data.shape[1]):
	s,p = scipy.stats.ranksums(htseq_data[Tcells,i],htseq_data[nonTcells,i])
	diffexp_ratio =  (numpy.mean(transcript_counts_logscale[Tcells,i]) - numpy.mean(transcript_counts_logscale[nonTcells,i]))*numpy.log2(10)
	diffexp_pvals.append(p)
	diffexp_ratios.append(diffexp_ratio)
diffexp_ratios = numpy.array(diffexp_ratios)	
diffexp_pvals = numpy.array(diffexp_pvals)
n = 100
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

diffexp_file = open(diffexp_filename,'w')
diffexp_file.write(('\t').join(('Gene','Wilcoxon rank-sum p-value','Log10 expression difference (T cells vs. epithelial)')) + '\n')
for i in range(len(genes_sorted)):
	if 'ensembl' in genes_sorted[i] or 'havana' in genes_sorted[i] or 'si:' in genes_sorted[i]:
		diffexp_file.write(genes_syst_sorted[i] + '\t' + str(pvals_sorted[i]) + '\t' + str(ratios_sorted[i]) + '\n')
	else:
		diffexp_file.write(genes_sorted[i] + '\t' + str(pvals_sorted[i]) + '\t' + str(ratios_sorted[i]) + '\n')
diffexp_file.close()

pt.figure()
pt.scatter(diffexp_ratios,-numpy.log10(diffexp_pvals),s=3)
ax = pt.gca()
for i in range(len(diffexp_inds)):
    ind = diffexp_inds[i]
    gene_name = diffexp_genes[i]
    if not (gene_name.startswith('si') or gene_name.startswith('ensembl') or gene_name.startswith('havana')):
        ax.text(diffexp_ratios[ind],-numpy.log10(diffexp_pvals)[ind],diffexp_genes[i],fontsize=4)
    #print(c1,c2,diffexp_ratios[ind],-numpy.log10(diffexp_pvals)[ind],diffexp_genes[i])
#pt.title('Differential expr)
ax.set_xlabel('Log2 fold change between population geometric means')
ax.set_ylabel(r'-Log$_{10}$p (Wilcoxon rank-sum test)')
pt.savefig('data_analysis_03_20_2020/diffexp_tcell_others.pdf')

###Rerun sam on the subset

htseq_data_tcells1 = htseq_data[Tcells,:]
htseq_indexes_tcells1 = htseq_indexes[Tcells]

samt = SAM(counts=(htseq_data_tcells1,htseq_headers,htseq_indexes_tcells1))
samt.preprocess_data(thresh=thresh) # log transforms and filters the data #sum_norm='cell_median',
samt.run(k=10) #run with default parameters	

####Look at batch effects based on plate

file = open(sample_list,'r')
snames = []
wells = []
pgroups = []
for line in file:
    well,sname = line.strip().split('\t')
    snames.append(sname)
    wells.append(well)
    if sname in  htseq_indexes_tcells1:
        if 'p1' in well:
            pgroups.append(0)

        elif 'p2' in well:
            pgroups.append(1)
        elif 'p3' in well:
            pgroups.append(2)
        elif 'p4' in well:
            pgroups.append(3)
file.close()

###dummy figure for legend
pt.figure()
ax = pt.gca()
symbol_list = []
for i in range(4):
    symb=ax.scatter(0,0,c=cdict[i],s=3)
    symbol_list.append(symb)

###actual scatter plot
pt.figure()
ax = pt.gca()

ax.scatter(samt.adata.obsm['X_umap'][:,0],samt.adata.obsm['X_umap'][:,1],c=[cdict[i] for i in pgroups],s=3)
#ax.scatter(standard_embedding[:,0],standard_embedding[:,1],c=[cdict[i] for i in pgroups],s=3)

ax.legend(symbol_list,['p1','p2','p3','p4'])
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')

pt.savefig('data_analysis_03_20_2020/sam_clustering_tcells_plates.pdf')
pt.close()


###Since there is a significant batch effect related to p1, repeat the analysis removing this plate
Tcells_to_check = set(htseq_indexes[Tcells])
htseq_data_new = []
htseq_indexes_new = []

for i in range(len(Tcells)):
    if pgroups[i] != 0:
        htseq_indexes_new.append(htseq_indexes_tcells1[i])
        htseq_data_new.append(htseq_data_tcells1[i,:])
        

htseq_data_tcells = numpy.array(htseq_data_new)
htseq_indexes_tcells = numpy.array(htseq_indexes_new)

###Rerun samalg

samt = SAM(counts=(htseq_data_tcells,htseq_headers,htseq_indexes_tcells))
samt.preprocess_data(thresh=thresh) # log transforms and filters the data #sum_norm='cell_median',
samt.run(k=10) #run with default parameters	

###Cluster

sam_clustert = cluster.KMeans(n_clusters=2,random_state=42).fit_predict(samt.adata.obsm['X_umap'])

###Make sure the bigger cluster is 0 and the other one is 1

bigclust,count = scipy.stats.mode(sam_clustert)
if bigclust[0] != 0:
    sam_clustert_swap = sam_clustert
    c1 = numpy.where(sam_clustert == 1)[0]
    c2 = numpy.where(sam_clustert == 0)[0]
    sam_clustert_swap[c1] = 0
    sam_clustert_swap[c2] = 1
    sam_clustert = sam_clustert_swap

###Repeat the differntial expression analysis between these two clusters; save a figure and file as above

###
inds_group1 = numpy.where(sam_clustert==0)[0]
inds_group2 = numpy.where(sam_clustert==1)[0]

transcript_counts_logscaleT = numpy.log10( htseq_data_tcells.T/numpy.sum(htseq_data_tcells,axis=1)*10**6 + 1).T


diffexp_pvals = []
diffexp_ratios = []
for i in range(htseq_data.shape[1]):
	s,p = scipy.stats.ranksums(htseq_data_tcells[inds_group1,i],htseq_data_tcells[inds_group2,i])
	diffexp_ratio =  (numpy.mean(transcript_counts_logscaleT[inds_group1,i]) - numpy.mean(transcript_counts_logscaleT[inds_group2,i]))*numpy.log2(10)
	diffexp_pvals.append(p)
	diffexp_ratios.append(diffexp_ratio)
diffexp_ratios = numpy.array(diffexp_ratios)	
diffexp_pvals = numpy.array(diffexp_pvals)
n = 100
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

diffexp_file = open(diffexp_filename_tcells,'w')
diffexp_file.write(('\t').join(('Gene','Wilcoxon rank-sum p-value','Log10 expression difference (T cells cluster 1 vs. T cells cluster 2)')) + '\n')
for i in range(len(genes_sorted)):
	if 'ensembl' in genes_sorted[i] or 'havana' in genes_sorted[i] or 'si:' in genes_sorted[i]:
		diffexp_file.write(genes_syst_sorted[i] + '\t' + str(pvals_sorted[i]) + '\t' + str(ratios_sorted[i]) + '\n')
	else:
		diffexp_file.write(genes_sorted[i] + '\t' + str(pvals_sorted[i]) + '\t' + str(ratios_sorted[i]) + '\n')
diffexp_file.close()

pt.figure()
pt.scatter(diffexp_ratios,-numpy.log10(diffexp_pvals),s=3)
ax = pt.gca()
for i in range(len(diffexp_inds)):
    ind = diffexp_inds[i]
    gene_name = diffexp_genes[i]
    if not (gene_name.startswith('si') or gene_name.startswith('ensembl') or gene_name.startswith('havana')):
        ax.text(diffexp_ratios[ind],-numpy.log10(diffexp_pvals)[ind],diffexp_genes[i],fontsize=4)
    #print(c1,c2,diffexp_ratios[ind],-numpy.log10(diffexp_pvals)[ind],diffexp_genes[i])
#pt.title('Differential expr)
ax.set_xlabel('Log2 fold change between population geometric means')
ax.set_ylabel(r'-Log$_{10}$p (Wilcoxon rank-sum test)')
pt.savefig('data_analysis_03_20_2020/diffexp_tcell_clusters.pdf')

###Display the clustering colors and a few marker genes

fig = pt.figure(figsize=(11,6))
gs = fig.add_gridspec(2,3,wspace=.3,hspace=.3)

ax = fig.add_subplot(gs[0,0])

ax.scatter(samt.adata.obsm['X_umap'][:,0],samt.adata.obsm['X_umap'][:,1],c=[cdict[s] for s in sam_clustert],s=6)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
gene_common_names = ['ptprc','trac','tagapa','lcp2a','srgn','arpc1b','coro1a','wasb','capgb','scinlb','pde4ba','junbb','icn','fa2h','ccl38.1']
ax.text(-.1,1.03,'A', transform=ax.transAxes)
gene_list = ['ENSDARG00000071437','ENSDARG00000075807','ENSDARG00000002353','ENSDARG00000055955','ENSDARG00000077069','ENSDARG00000027063','ENSDARG00000054610','ENSDARG00000026350','ENSDARG00000099672','ENSDARG00000058348','ENSDARG00000032868','ENSDARG00000104773','ENSDARG00000009978','ENSDARG00000090063','ENSDARG00000041919']

htseq_data_tcells_norm = (htseq_data_tcells.T/numpy.sum(htseq_data_tcells,axis=1)).T*10**6 ####Counts per million

###Construct a dataframe with expression vectors for the chosen genes, as well as the cluster assignment categorical variable
htseq_data_tcells_norm_log = numpy.log10(htseq_data_tcells_norm + 1)
exp_list = []
for i in range(len(gene_list)):
    gene = gene_list[i]
    exp = utilities.get_gene_expression(htseq_headers,htseq_data_tcells_norm_log,gene)
    exp_list.append(exp)

#exp_list.append(sam_clustert)
#col_keys = gene_common_names
#col_keys.append('cluster_assignment')

exp_list = numpy.array(exp_list).T

    
df = pd.DataFrame(exp_list,columns=["GN"+str(g) for g in range(len(gene_common_names))],index=htseq_indexes_tcells)
print(df)
df["id"] = df.index
df["cluster_assignment"] = sam_clustert
df_long = pd.wide_to_long(df,["GN"],i="id",j="gene")

df_long["gene_exp"] = df_long.index.get_level_values("gene")

print(df_long)

ax = fig.add_subplot(gs[0,1:])
#ax.set_yscale("log")
sns.violinplot(x="gene_exp",y="GN",hue="cluster_assignment",data=df_long,ax=ax,legend=False,order=[0,'',1,2,3,4,'',5,6,7,8,9,'',10,11,'',12,13,14],cut=0,linewidth=.5,scale='width')

ax.set_xticks([0,2,3,4,5,7,8,9,10,11,13,14,16,17,18])
ax.text(0,5,'Immune',horizontalalignment='center',verticalalignment='top',fontsize=8)
ax.text(3.5,5,'T cell',horizontalalignment='center',verticalalignment='top',fontsize=8)
ax.text(9,5,'Actin/motility',horizontalalignment='center',verticalalignment='top',fontsize=8)
ax.text(13.5,5,'Cluster 1',horizontalalignment='center',verticalalignment='top',fontsize=8)
ax.text(17,5,'Cluster 2',horizontalalignment='center',verticalalignment='top',fontsize=8)

ax.axvline(1,0,1,color='k',linewidth=.5)
ax.axvline(6,0,1,color='k',linewidth=.5)
ax.axvline(12,0,1,color='k',linewidth=.5)
ax.axvline(15,0,1,color='k',linewidth=.5)

ax.set_xlim(-1,19)
ax.set_ylim(-.5,5.2)
ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels([r'$10^0$',r'$10^1$',r'$10^2$',r'$10^3$',r'$10^4$'])
ax.set_xticklabels(gene_common_names,fontsize=9,rotation=90)
ax.set_xlabel('')
ax.get_legend().remove()
ax.set_ylabel(r'CPM+1')
ax.text(-.1,1.03,'B', transform=ax.transAxes)
###Find the top correlates of arpc1b
exp_genes = numpy.sum(htseq_data_tcells > .5,axis=0)/htseq_data_tcells.shape[0] > .2
htseq_headers_exp = htseq_headers[exp_genes]
htseq_headers_trans_exp = htseq_headers_translated[exp_genes]
htseq_data_tcells_exp = htseq_data_tcells_norm[:,exp_genes]

ngenes = len(htseq_headers_exp)
arpc1b_exp = utilities.get_gene_expression(htseq_headers,htseq_data_tcells_norm,'ENSDARG00000027063')
arpc1b_corrs = []
arpc1b_ps = []
for i in range(ngenes):
    ##Take out cells that drop out in both
    included_cells = numpy.logical_or(htseq_data_tcells_exp[:,i]>.001,arpc1b_exp>.001)
    corr,p = scipy.stats.spearmanr(htseq_data_tcells_exp[included_cells,i],arpc1b_exp[included_cells])
    arpc1b_corrs.append(corr)
    arpc1b_ps.append(p)

corr_ranks = numpy.argsort(arpc1b_corrs)[::-1]
ntop = 4

arpc1b_ps = numpy.array(arpc1b_ps)

hist_bins = numpy.arange(-.4,.4,.02)
arpc1b_corrs = numpy.array(arpc1b_corrs)
ax = fig.add_subplot(gs[1,0])
sns.distplot(arpc1b_corrs[corr_ranks[1:]],ax=ax,color='grey')
ax.set_xlabel(r'Spearman $\rho$ stat with arpc1b')
###Label top hits

ntop = 4
arpc1b_norm = numpy.log10(arpc1b_exp+1)
exp_module_mat = []

pval_labels = [r'p$<10^{-3}$',r'p$<10^{-3}$',r'p=.002',r'p=.002']
legend_label_list = []
legend_symbol_list = []
clist = ['C2','C3','C4','C5','C6','C7','C8']
for i in range(ntop):
    counter = i+1
    ind = corr_ranks[counter]
    gene_name = htseq_headers_trans_exp[ind]
    if gene_name.startswith('si'):
        gene_name = 'BX511128.3'
    #c = i%nc
    #r = int(numpy.floor(i/nc))
    included_cells = numpy.logical_or(htseq_data_tcells_exp[:,ind]>.001,arpc1b_exp>.001)
    corr,p = scipy.stats.spearmanr(htseq_data_tcells_exp[included_cells,ind],arpc1b_exp[included_cells])
    #ax.text(corr,.5,gene_name + ', ' + pval_labels[i],rotation=90,horizontalalignment='center',fontsize=6)
    legend_label_list.append(gene_name + '\n' + pval_labels[i])
    
    l=ax.axvline(corr,0,.05,color=clist[i])
    legend_symbol_list.append(l)
    #if counter < 3.5:
    #    ax.text(corr,.5,gene_name,rotation=90,horizontalalignment='center',fontsize=6)
    #elif counter > 4.5:
     #   ax.text(corr,.5,prev_gene_name+ ',' + gene_name,rotation=90,horizontalalignment='center',fontsize=6)
    
    prev_gene_name = gene_name
    exp_module_mat.append(numpy.log10(htseq_data_tcells_exp[:,ind]+1))

#nbott = 1
#for i in range(nbott):
#    counter = i-1
 #   ind = corr_ranks[counter]
 #   gene_name = htseq_headers_trans_exp[ind]
 #   if gene_name.startswith('si'):
 #       gene_name = htseq_headers_exp[ind]
    #c = i%nc
    #r = int(numpy.floor(i/nc))
 #   included_cells = numpy.logical_or(htseq_data_tcells_exp[:,ind]>.001,arpc1b_exp>.001)
 #   corr,p = scipy.stats.spearmanr(htseq_data_tcells_exp[included_cells,ind],arpc1b_exp[included_cells])
    #ax.text(corr,.5,gene_name + ', ' + pval_labels[i],rotation=90,horizontalalignment='center',fontsize=6)
 #   legend_label_list.append(gene_name)# + '\n' + pval_labels[i])
    
 #   l=ax.axvline(corr,0,.05,color=clist[i+ntop])
 #   legend_symbol_list.append(l)
    #if counter < 3.5:
    #    ax.text(corr,.5,gene_name,rotation=90,horizontalalignment='center',fontsize=6)
    #elif counter > 4.5:
     #   ax.text(corr,.5,prev_gene_name+ ',' + gene_name,rotation=90,horizontalalignment='center',fontsize=6)
    
 #   prev_gene_name = gene_name
 #   exp_module_mat.append(numpy.log10(htseq_data_tcells_exp[:,ind]+1))
ax.legend(legend_symbol_list,legend_label_list, fontsize=8)
ax.set_xlim(-.45,.49)
####Plot example correlation to show overlap between two clusters
ax.text(-.1,1.03,'C', transform=ax.transAxes)
exp_module_mat = numpy.array(exp_module_mat).T
reg = LinearRegression().fit(exp_module_mat, arpc1b_norm)

arpc1b_exp = utilities.get_gene_expression(htseq_headers,htseq_data_tcells,'ENSDARG00000027063')
actb2_exp = utilities.get_gene_expression(htseq_headers,htseq_data_tcells,'ENSDARG00000037870')
cdc42l_exp = utilities.get_gene_expression(htseq_headers,htseq_data_tcells,'ENSDARG00000040158')

ax = fig.add_subplot(gs[1,1])
ind = corr_ranks[1] ###top hit after arpc1b itself
#actb2_exp = htseq_data_tcells_exp[:,ind]
bigclust,count = scipy.stats.mode(sam_clustert)
#print(bigclust)
l1 = sam_clustert == bigclust[0]
l2 = sam_clustert != bigclust[0]
ax.set_yscale('log')
ax.set_xscale('log')

ax.scatter(actb2_exp[l1]+1,arpc1b_exp[l1]+1,c=[cdict[cl] for cl in sam_clustert[l1]],zorder=0,alpha=.8,s=6)
ax.scatter(actb2_exp[l2]+1,arpc1b_exp[l2]+1,c=[cdict[cl] for cl in sam_clustert[l2]],zorder=0,alpha=.8,s=6)
ax.set_ylim([.8,5*10**4])
ax.set_xlim([.8,10**6])
ax.set_ylabel('arpc1b expression')
ax.set_xlabel('actb2 expression')
ax.text(-.1,1.03,'D', transform=ax.transAxes)
ax = fig.add_subplot(gs[1,2])
ind = corr_ranks[1] ###top hit after arpc1b itself
#actb2_exp = htseq_data_tcells_exp[:,ind]
bigclust,count = scipy.stats.mode(sam_clustert)
#print(bigclust)
l1 = sam_clustert == bigclust[0]
l2 = sam_clustert != bigclust[0]
ax.set_yscale('log')
ax.set_xscale('log')

ax.scatter(cdc42l_exp[l1]+1,arpc1b_exp[l1]+1,c=[cdict[cl] for cl in sam_clustert[l1]],zorder=0,alpha=.8,s=6)
ax.scatter(cdc42l_exp[l2]+1,arpc1b_exp[l2]+1,c=[cdict[cl] for cl in sam_clustert[l2]],zorder=0,alpha=.8,s=6)
ax.set_ylim([.8,5*10**4])
ax.set_xlim([.8,5*10**4])
#ax.set_ylabel('arpc1b expression')
ax.set_xlabel('cdc42l expression')
ax.text(-.1,1.03,'E', transform=ax.transAxes)
pt.savefig('data_analysis_03_20_2020/Tcell_corrs_multipanel_trial.pdf',bbox_inches='tight')
pt.close()