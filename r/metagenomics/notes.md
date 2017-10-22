# Notes on metagenomics

Attached are 4 R scripts designed to be run on a project at the end of the pipeline. They are currently tuned to work on version 3 and will need modifying for each new pipeline version. They are currently fairly minimal and will probably want a bit of tweaking before deploying in production. It would obviously be convenient to have a way of jointly developing these scripts. If you want, I can create a little public github repo to put them in, or you could incorporate them into your repo and give me read access to it so I can submit pull requests, or whatever. Note that I'm not a big graphics person, so if you need fancy colourful ggplot graphics or interactive javascript plots, I'll need some help on that.

All scripts are designed to be run (only) from a project directory, which is assumed to have a directory called "version_3.0" as a subdirectory. They can easily be tweaked for starting from other directories. They produce TSV and SVG files which you could pick up from your web front end in some way.

The scripts are as follows:

diversity.R
This script relies on the latest (rforge) version of my CRAN package "ebimetagenomics" which contains all of the actual functionality used by this script. This can be installed with:
install.packages("ebimetagenomics", repos="http://R-Forge.R-project.org")
It relies on a handful of other CRAN packages which you might need to hand-install. Note that you really do need version 0.3 of this package (on rforge), not the version 0.2 which is on CRAN. Once we are happy that this script functions as desired, I'll push 0.3 to CRAN, which will make setup a bit simpler.
The script walks through the run directories and computes a load of (alpha) diversity and coverage measures. Some simple taxa abundance distribution plots are stored in each "charts" subdirectory, and the statistics are accumulated into a TSV file stored in the "project-summary" directory. This summary file contains various estimates of the total number of species in the sampled population and estimates of the sequencing effort required in order to observe a given fraction of the total species present.

comparisons.R
This script uses the taxa summaries in the project-summary directory to make comparisons between the different runs of the project. It relies heavily on the Bioconductor package "phyloseq". Unfortunately the installation of Bioconductor packages often isn't quite as easy as for CRAN packages, but it worked OK for me once I upgraded my R installation to the latest version. Some install instructions can be found here:
https://joey711.github.io/phyloseq/install.html
Currently the only saved output for this script is a bunch of plots. We can also store some quantitative information if we decide on what we want and how we want to store it. We might need a skype chat about that at some point. First off a simple PCA plot is produced, based on proportions of hits on each taxa. Actually this doesn't rely on phyloseq, but it is a useful diagnostic plot for identifying outlying runs.
The more interesting output is a heatmap for each run (stored in the charts directory), which illustrates the fold-change between this run and every other run in the project. Note that here "fold-change" is estimated robustly by phyloseq in a way that shouldn't be sensitive to library size and does not rely on rareification.

The above scripts are run-based, and don't rely on knowing anything about the mapping between "sample" and "run". For projects with a one-one mapping between sample and run this obviously doesn't matter, but for other projects, "run" is really not the right level to be looking at these kinds of analyses - they should all be computed at the sample level. However, you don't currently have information on the mapping stored in your version 3 results directory, making it tricky for me to use that info. Alex has sent me an example of a mapping file which could be stored in the project-summary directory. The following 2 scripts assume that such a file is present.

diversity-sample.R
This is an alternative version of the diversity.R script which pools together all of the runs for a sample before computing statistics. All output is stored in the project-summary directory, since your folder structure doesn't reflect samples.

comparisons-sample.R
This is an alternative version of the comparisons.R script which uses sample information. Here the PCA plot is identical except that runs are coloured according to sample, which makes it much more useful as a diagnostic - eg. to spot runs which are different to other runs in a sample. The heatmaps are also based on samples rather than runs. This is important not only because we are typically more interested in differences between samples than runs, but also because it is easier to more robustly estimate fold-chance when within-sample replication information is available.

# Notes


This directory mainly consists of R scripts for metagenomics analysis, for potential integration into the EBI metagenomics pipeline.

Also tests for the new metagenomics API in `api-test.R` and `rjsonapi-test.R`

## Interesting projects

OSD: ERP009703

Tara:
* ERP007024 - Amplicon RNA protoists
* ERP001736 - Shotgun prokaryotes
* ERP006153 - Amplicon DNA virus
* ERP003634 - Amplicon prokaryotes/protoists

Microbiome QC project: SRP047083

Sample project from Alex: DRP003216


#### eof

