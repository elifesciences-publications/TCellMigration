from glob import glob

"""
Snakefile for basic zcat operations on gzipped fastqs and generation of directories.
Also includes paths to common scripts
"""


RESOURCES =                 '/oak/stanford/groups/quake/ejerison/Resources'
GENOMEDIR =				    RESOURCES + '/Star_dbs/zebrafish'
REFERENCE_ANNOTATION =		RESOURCES + '/zebrafish_reference_genome/Danio_rerio.GRCz10.85.ERCC.GFP.gtf'
STAR =						'/home/users/ejerison/STAR/bin/Linux_x86_64/STAR'
BOWTIE2 = '/home/users/ejerison/miniconda3/bin/bowtie2'
BASIC = '/home/users/ejerison/miniconda3/pkgs/basic-1.4.1-py_0/python-scripts/BASIC.py' ###Note that BASIC must be called from the pkgs folder, or else it is unable to find its auxiliary databases
#FASTQ_DIR =					'/oak/stanford/groups/quake/ejerison/NovaSeq/zHSCMacro022718'
#WORKING_DIR =				'/oak/stanford/groups/quake/ejerison/single_cell_analysis/HSC_Macrophage_4_2_2018'
FASTQ_DIR =					'/oak/stanford/groups/quake/ejerison/NovaSeq/lckgfp_20190214'
WORKING_DIR =				'/oak/stanford/groups/quake/ejerison/single_cell_analysis/Tcells_02_20_2019'
SAMTOOLS =					'/home/users/ejerison/miniconda3/bin/samtools'
# -----------------------------------
# Function for loading seeds


def load_seeds(infile):
    seeds = []
    with open(infile, 'rU') as f:
        for line in f:
            seeds.append(line.rstrip())
    return seeds


def load_seeds_with_basenames(infile):
    seeds = []
    basenames = []
    with open(infile, 'rU') as f:
        for line in f:
            vals = line.rstrip().split("\t")
            seeds.append(vals[0])
            basenames.append(vals[1])
    return seeds, basenames

def get_sample_names(sample_dir):
	
	folders = glob( sample_dir + '/*')
	
	samples = []
	
	for folder in folders:
		
		sample = folder.split('/')[-1]
		
		samples.append( sample )
	
	return samples

def get_sample_basenames():
	parent_dir = FASTQ_DIR
	samples_R1 = glob( parent_dir + '/*R1_001.fastq.gz')
	sample_basenames = []
	for sample in samples_R1:
		sample_basenames.append( ('_').join(sample.split('/')[-1].split('_')[0:-2]) )
	
	return sample_basenames

def get_R1_R2_from_basename(wildcards):
	
	parent_dir = FASTQ_DIR
	sample_basename = wildcards.sample
	
	R1_R2 = [parent_dir + '/' + sample_basename + '_R1_001.fastq.gz', parent_dir + '/' + sample_basename + '_R2_001.fastq.gz']
	
	return R1_R2
# -----------------------------------
# Functions for transferring files to/from cluster


def name_on_scratch(s, scratch):
    return scratch+"/"+os.path.basename(s)


def names_on_scratch(names, scratch):
    return [name_on_scratch(n, scratch) for n in names]


def cp_to_scratch(inputs, scratch):
    for i in inputs:
      cmd = "rsync -aW " + i + " " + name_on_scratch(i, scratch)
      subprocess.call(cmd, shell=True)
    return None


def cp_from_scratch(outputs, scratch):
    for o in outputs:
        cmd = "rsync -aW " + name_on_scratch(o, scratch) + " " + o
        subprocess.call(cmd, shell=True)
    return None

# -----------------------------------
# Functions for manipulating multiplq fastq files


def get_all_files(d):
    return [d+"/"+f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]

def get_R1_R2(wildcards):
	parent_dir = FASTQ_DIR
	sample_dir = wildcards.sample
	R1_R2_list = [glob(parent_dir + '/' + sample_dir + '/*R1*.gz')[0], glob(parent_dir + '/' + sample_dir + '/*R2*.gz')[0]]
	return R1_R2_list

def get_all_fastq_gzs_R1(wildcards):
    all_files = get_all_files(wildcards.parent_dir)
    fastq_gzs = []
    for f in all_files:
        if ("L00" in f) and ("_R1_" in f) and (".fastq" in f) and (os.path.splitext(f)[1] == ".gz"):
            fastq_gzs.append(f)
    return sorted(fastq_gzs)


def get_all_fastq_gzs_R2(wildcards):
    all_files = get_all_files(wildcards.parent_dir)
    fastq_gzs = []
    for f in all_files:
        if ("L00" in f) and ("_R2_" in f) and (".fastq" in f) and (os.path.splitext(f)[1] == ".gz"):
            fastq_gzs.append(f)
    return sorted(fastq_gzs)


def get_all_fastq_gzs_R1_v2(wildcards):
    all_files = get_all_files(wildcards.dir+ '/' + wildcards.basename)
    fastq_gzs = []
    for f in all_files:
        if ("00" in f) and ("_R1_" in f) and (".fastq" in f) and (os.path.splitext(f)[1] == ".gz"):
            fastq_gzs.append(f)
    return sorted(fastq_gzs)


def get_all_fastq_gzs_R2_v2(wildcards):
    all_files = get_all_files(wildcards.dir + '/' + wildcards.basename)
    fastq_gzs = []
    for f in all_files:
        if ("00" in f) and ("_R2_" in f) and (".fastq" in f) and (os.path.splitext(f)[1] == ".gz"):
            fastq_gzs.append(f)
    return sorted(fastq_gzs)


def unzip_fastq(f):
    cmd = "gunzip " + f
    p = subprocess.Popen(cmd, shell=True)
    return None


# -----------------------------------
# Common rules

rule zcat_R1:
  """ Concatenate fastq files """
  input: get_all_fastq_gzs_R1
  output: '{parent_dir}/{sample}/{sample}.Combined_R1.fastq'
  params: name='zcat', partition="general", mem="5300"
  shell: 'zcat {input} > {output}' 
  
rule zcat_R2:
  input: get_all_fastq_gzs_R2
  output: '{parent_dir}/{sample}/{sample}.Combined_R2.fastq'
  params: name='zcat', partition="general", mem="5300"
  shell: 'zcat {input} > {output}'

rule zcat_R1_v2:
  """ Concatenate fastq files """
  input: get_all_fastq_gzs_R1_v2
  output: '{dir}/{basename}/{basename}.Combined_R1.fastq'
  params: name='zcat', partition="general", mem="5300"
  shell: 'zcat {input} > {output}'

rule zcat_R2_v2:
  input: get_all_fastq_gzs_R2_v2
  output: '{dir}/{basename}/{basename}.Combined_R2.fastq'
  params: name='zcat', partition="general", mem="5300"
  shell: 'zcat {input} > {output}'

rule mkdir:
  """ Make directories """
  output: '{dir}/mkdir.time'
  params: name='mkdir', partition='general', mem='1024'
  run:
    dir, _ = os.path.split(output[0])
    shell('mkdir -p {dir}/exon_output &&\
           mkdir -p {dir}/fastqc_output &&\
           mkdir -p {dir}/htseq_output &&\
           mkdir -p {dir}/preprocess_fastq &&\
           # mkdir -p {dir}/prinseq_output &&\
           mkdir -p {dir}/qc_output &&\
           mkdir -p {dir}/STAR_output &&\
           # mkdir -p {dir}/barcode_output &&\
           mkdir -p {dir}/tmp &&\
           touch {output[0]}')
