import os
import sys

include: "Snakefile_accessory_tcell_v2.py"


__author__ = 'Elizabeth Jerison'
__date__ = '15/10/21'
__last_updated__ = '19/02/20'

# where rule logfiles will be output
workdir: '/oak/stanford/groups/quake/ejerison/single_cell_analysis/Tcells_02_20_2019/snakemake_logs'


# -----------------------------------
# Read seedfile

#SEEDFILE = '/local10G/ejerison/singlecell/zebrafish_bcells_11_7_2016/sample_folder_Derek'

#SEEDFILE = '/local10G/ejerison/singlecell/HSC_RNA_seq/HSC_seed_file.txt'

#SEEDFILE = '/local10G/dcroote/singlecell/fastq/151027_IgEseq1/folder_structure1.txt'

# Format (tab separated):
# parent_dir	sample_folder
# Example:
# /local10G/dcroote/singlecell/nextseq_run_1	cell_in_well_A1
# /local10G/dcroote/singlecell/nextseq_run_1	cell_in_well_A2

#PARENT_DIRS, SAMPLES = load_seeds_with_basenames(SEEDFILE)

# -----------------------------------
# Parameters and executables:
# see Snakefile_accessory.py


# -----------------------------------
# Rules
SAMPLES = get_sample_basenames()
print(SAMPLES)

rule all:
  input: expand('{parent_dir}/{sample}/htseq_output/htseq.tsv',parent_dir=WORKING_DIR, sample=SAMPLES)
    #expand('{parent_dir}/{sample}/BASIC_output/tcr_assembly.fasta', parent_dir=WORKING_DIR, sample=SAMPLES),
  params: name='all', partition='quake,owners', mem='1024'

#To additionally call isotypes, change the first argument of expand to the output of 'call_isotype'

rule star:
  """ Map reads to genome using STAR
      Uses zcat rules to combine gzipped fastqs as input
      (may not be necessary now that fastq files are
      not split by lane)
  """
  input: get_R1_R2_from_basename
  output: '{parent_dir}/{sample}/STAR_output/Aligned.out.bam'
  params: name='star', partition='quake,owners', mem='32000',time='04:00:00'
  threads: 6
  run:
    wdir = os.path.dirname(str(output))
    shell("mkdir -p {wdir} && "
          "{STAR} "
          "--genomeDir {GENOMEDIR} "
          "--readFilesIn {input[0]} {input[1]} "
          "--readFilesCommand gunzip -c "
          "--outFilterMismatchNoverReadLmax 0.04 "
          "--outFilterMatchNminOverLread 0.4 "
          "--outFilterScoreMinOverLread 0.4 "
          "--outSAMstrandField intronMotif "
          "--runThreadN {threads} "
          "--outSAMtype BAM Unsorted "
          "--outFileNamePrefix {wdir}/ " 
          "--outReadsUnmapped Fastx")

    # CHANGE: GENOMEDIR to desired STAR genome directory
    # outFilterMismatchNoverLmax: default=0.3

rule sort:
   """ Re-sort STAR output by name """
    input: rules.star.output
    output: '{parent_dir}/{sample}/STAR_output/Aligned_SortedByName.out.bam'
    params: name='sort', partition='quake,owners', mem='32000',time='04:00:00'
    threads: 6
    shell: "{SAMTOOLS} sort -n -o {output} -@ 6 "
           "{input}"
rule htseq:
  """ Count reads mapping to features using htseq """
  input:  rules.sort.output
  output: '{parent_dir}/{sample}/htseq_output/htseq.tsv'
  params: name='htseq', partition='quake,owners', mem='32000',time='24:00:00'
  threads: 6
  run:
    wdir = os.path.dirname(str(output))
    shell("mkdir -p {wdir} && "
           "htseq-count -s no -r name -f bam -m intersection-nonempty --nonunique all "
           "{input} {REFERENCE_ANNOTATION} > {output}")

rule basic:
	input: get_R1_R2_from_basename
	output: '{parent_dir}/{sample}/BASIC_output/tcr_assembly.fasta'
	params: name='basic', partition='quake,owners', mem='32000',time='24:00:00'
	threads: 6
	run: 
		wdir = os.path.dirname(str(output))
		shell("mkdir -p {wdir} && "
			   "python {BASIC} -i TCR -p 6 -n tcr_assembly -PE_1 {input[0]} -PE_2 {input[1]} -g drerio -b {BOWTIE2} -t {wdir} -o {wdir} -a -v")
		