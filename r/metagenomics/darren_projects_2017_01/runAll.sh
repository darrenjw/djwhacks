#!/bin/bash
# runAll.sh
# run scripts over all directories

echo
echo "Starting script now"
echo

for name in ??P??????
do
    cd $name
    echo
    echo "*********************************************************"
    echo "*********************************************************"
    echo "*********************************************************"
    echo
    echo "$name"
    echo
    date
    echo
    echo
    # Rscript ~/src/git/djwhacks/r/metagenomics/comparisons.R
    # Rscript ~/src/git/ebi-metagenomics-stats/comparison/comparisons.R
    Rscript ~/src/git/djwhacks/r/metagenomics/diversity.R
    # Rscript ~/src/git/ebi-metagenomics-stats/comparison/diversity.R
    cd ..
done

echo
echo "Script finished."
echo
date
echo
echo

# eof

