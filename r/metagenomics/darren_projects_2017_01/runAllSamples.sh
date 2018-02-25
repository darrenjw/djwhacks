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
    Rscript ~/src/git/djwhacks/r/metagenomics/comparisons-sample.R
    cd ..
done

echo
echo "Script finished."
echo
date
echo
echo

# eof

