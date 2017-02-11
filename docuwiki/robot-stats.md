###### Robot Stats

#### Statistical analysis of the data from ROD

### People

` * `[`Darren` `Wilkinson`](people:darren "wikilink")\
` * `[`Steve` `Addinall`](people:steve "wikilink")\
` * `[`Jonathan` `Heydari`](people:jonathan "wikilink")\
` * `[`Conor` `Lawless`](people:conor "wikilink")\
` * `[`David` `Lydall`](people:david "wikilink")\
` * `[`Alex` `Young`](people:alexyoung "wikilink")\
` * `[`Yu` `Jiang`](people:yujiang "wikilink")

##### Basic Robot facts

` * Robot math:`\
`   * 8*12=96, 16*24=384, 32*48=1536 (rows*cols)`\
`   * 96*4=384, 384*4=1,536 (plate to plate)`\
`   * 2*(16+22)=76 spots around the edge of a 384 plate`\
`   * 14*22=384-76=308 spots not on the edge of a 384 plate`\
` * There are around 4,600 (actually 4,300?) strains in the full plate library (out of around 6,000 genes) - must be (nearly) all of the non-essential genes.`\
` * Full "Boone" library on 14 plates. 384*14=5,376 spotids, but only 4,293 distinct ORFs.`\
` * 16 special ORFs (occurring exactly twice in the library):`\
`   * `[`YBL103C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBL103C "wikilink")`  (RTG3) 1-13-11+14-12-02 (plate 1, row 13, col 11 and plate 14, row 12, col 2)`\
`   * `[`YBR020W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBR020W "wikilink")` (GAL1) 1-02-18+14-10-22`\
`   * `[`YBR061C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBR061C "wikilink")` (TRM7) 1-08-20+14-10-20`\
`   * `[`YBR075W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBR075W "wikilink")` (inside `[`YBR074W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBR074W "wikilink")`) 1-10-16+14-10-18`\
`   * `[`YBR082C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBR082C "wikilink")`  (UBC4) 1-10-10+14-10-16`\
`   * `[`YBR095C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBR095C "wikilink")`  (RXT2) 1-12-14+14-10-10`\
`   * `[`YBR115C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBR115C "wikilink")`  (LYS2) 1-14-08+14-12-22`\
`   * `[`YBR165W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YBR165W "wikilink")`  (UBS1) 1-06-09+14-12-18`\
`   * `[`YDL074C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YDL074C "wikilink")`  (BRE1) 2-14-06+14-12-12`\
`   * `[`YDR483W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YDR483W "wikilink")`  (KRE2) 4-09-18+14-14-10`\
`   * `[`YJR109C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YJR109C "wikilink")`  (CPA2) 8-03-12+11-07-11`\
`   * `[`YLR455W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YLR455W "wikilink")`  (no name) 10-09-22+14-08-03`\
`   * `[`YMR119W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YMR119W "wikilink")`  (ASI1) 10-10-02+14-10-05`\
`   * `[`YOR300W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YOR300W "wikilink")`  (overlaps `[`YOR299W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YOR299W "wikilink")`, BUD7) 13-13-02+14-12-21`\
`   * `[`YOR306C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YOR306C "wikilink")`  (MCH5) 13-15-12+14-12-19`\
`   * `[`YOR309C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YOR309C "wikilink")`  (overlaps `[`YOR310C`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YOR310C "wikilink")`, NOP58) 13-15-06+14-12-17`\
` * `

<html>
Excerpt from e-mail from Renee Brost in Charlie Boone's laboratory
<font color=purple>"In all cases with the exception of CPA2, you may
have noticed that the second copy of these duplicate genes are on Plate
14. They were included to ensure plate 14 was completely filled since
empty spaces can cause noise in our analysis. Genes for which there are
duplicates were randomly chosen by Amy Tong. I believe the addition of
CPA2 twice was likely just an accidental duplication that occurred
during array construction."</font>

</html>
` * Also `[`YOR202W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YOR202W "wikilink")` (His3) - around the edge of each plate. This occurs 1,068 times in the library, but 76x14=1,064, so round the edge of each plate and then 4 other places in the library (13-03-20, 14-15-03, 14-15-05, 14-15-07). These 4 His3 spots can be used as a reasonable "control", since the yeast strains actually have another copy of the His3 gene elsewhere in their genome (it's all part of the way that the synthetic genetic array (SGA) procedure works).`\
` * Discounting "special" orfs, there are 4,276 "regular" orfs (occurring exactly once). Total spotids is: 4,276+16x2+1,068=5,376`\
` * 308*14=4,312 spots not around the edge in the 14 plate library`\
` * **Now plate 15** - mainly containing repeats of strains already in the library - very useful for checking consistency, etc. However, one new knockout introduced on plate 15 - `[`YNL250W`](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=YNL250W "wikilink")` (Rad50) - a Boone "slow-grower" - so 4,294 distinct ORFs in the 15 plate version of the library`\
` * Now also **version 3** of the library (Boone library referred to as version 2) - same spots on each plate, but re-arranged, to help correct for location effects`\
` * Images in cisbclust:/data/rod/Images (`*[`smb:cisbclust/rod`](smb:cisbclust/rod)*`)`\
` * Subversion path for ROD stats: `[`http://metagenome.ncl.ac.uk/subversion/cisban/robot-stats/trunk`](http://metagenome.ncl.ac.uk/subversion/cisban/robot-stats/trunk)\
` * Some `[`example`
`images`](http://picasaweb.google.com/lydall.robots/CONT1280120081418?authkey=BNJk6IlkRRI "wikilink")` are available as part of the `[`Lydall`
`lab` `public`
`gallery`](http://picasaweb.google.com/lydall.robots "wikilink")` on `[`picasa`](http://picasaweb.google.com/home "wikilink")\
` * A "robot stats starter pack" is available as a `` for anyone wanting to have a play with the R code. Just do 'source("yku70.R")' at the R command prompt. You might need to install a couple of R packages, but it will tell you which ones, and is very easy. eg. 'install.packages("car")'.`\
` * Subversion directory for the latest hitlists: `[`http://metagenome.ncl.ac.uk/subversion/cisban/robot-stats/trunk/hitlists`](http://metagenome.ncl.ac.uk/subversion/cisban/robot-stats/trunk/hitlists)` `\
` * `[`Robot` `Screen` `Image`
`Browser`](https://basis1.ncl.ac.uk/basis/Robot/ "wikilink")

##### Relevant papers, software, etc.

Papers
------

` * `` - paper discussing the notion of genetic interaction, fundamental to these screens`\
` * `` - from classical genetics to quantitative genetics to systems biology: Modeling Epistasis`\
` * `` - review of epistatis in Nat Rev Gen`\
` * `` - Functional dissection of protein complexes involved in yeast chromosome biology using a genetic interaction map`

Software
--------

` * `[`R`](http://www.r-project.org/ "wikilink")` - the statistical programming language`\
`   * `[`BioConductoR`](http://www.bioconductor.org/ "wikilink")` - suite of R packages for bioinformatics`\
`     * `[`SLGI`](http://bioconductor.org/packages/2.2/bioc/html/SLGI.html "wikilink")` - synthetic lethal genetic interactions (BioC package)`

Other
-----

` * `[`Tyers` `lab` `stuff` `on`
`SGA`](http://tyerslab.bio.ed.ac.uk/wikiworld/sga/index.php/Synthetic_Genetic_Analysis "wikilink")\
` * Darren's talk on `` for CISBAN altogether`\
` * Darren's `\
` * Darren's `\
` * Our `[`discussion` `about` `hit`
`scoring`](robot:discussion_about_hit_scoring "wikilink")` is also very relevant...`\
` * `[`Interesting` `yeast`
`genes`](robot:interesting "wikilink")` - genes to include/look out for in analyses`\
` * Darren's `[`introduction` `to` `multiple`
`testing`](multipletesting "wikilink")\
` * Darren's guide to `[`testing` `for`
`over-representation`](overrep "wikilink")` (hyper-geometric test)`

##### Experiments

` * `[`CDC13-1` `partial` `screen`](robot:cdc13-1_Steve "wikilink")\
` * `[`YKU70` `deletion` `full` `genome`
`screen`](robot:yku70 "wikilink")\
` * `[`TS` `and`
`Up-Down`](robot:temperature_sensitive_and_up-down_controls "wikilink")\
` * `[`CDC13-1` `full` `screen`](robot:cdc13-1_Kaye "wikilink")\
` * `[`TMPyP4` `screen`](robot:tmpyp4 "wikilink")\
` * `[`CDC13-1` `1536/384` `comparison`](robot:cdc13-1-1536 "wikilink")\
` * `[`TS/UD` `control` `screen`](robot:control "wikilink")\
` * `[`Passage` `timecourse`](robot:passage "wikilink")\
` * `[`Pombe` `zinc` `arsenite`](robot:pombe "wikilink")\
` * `[`cdc13-1` `rad9Δ` `screen`](robot:rad9 "wikilink")\
` * `[`Full` `genome` `1536` `passage` `time`
`course`](robot:passage-full "wikilink")\
` * `[`EST1-G1` `passage`
`experiment`](robot:passage-est1-g1 "wikilink")\
` * `[`Steve's` `new` `YKU70` `screen`](robot:yku70-steve "wikilink")\
` * `[`Amanda's` `cdc13-1` `screen` `minus`
`edges`](robot:amanda "wikilink")\
` * `[`Steve's` `new` `single` `deletion` `control`
`screen`](robot:sdl "wikilink")\
` * `[`Steve's` `YKU70` `experiment` `in`
`1536`](robot:yku1536 "wikilink")\
` * `[`Steve's` `SDL` `control` `in` `1536`](robot:sdl1536 "wikilink")\
` * `[`Control` `SGA` `screen`](robot:csga "wikilink")\
` * `[`Adam's` `new` `CDC13-1` `screen`](robot:cdc131-new "wikilink")\
` * `[`Adam's` `Overexpression` `library`](yeast2:OELib "wikilink")\
` * `[`Adam's` `bir1-17` `screen`](yeast2:bir1-17 "wikilink")\
` * `[`Greg's` `exo1D` `cdc13-1` `xyzD`
`screen`](yeast2:exo1cdc13 "wikilink")

##### Meta-analyses

` * `[`CDC131-1/YKU70` `Meta-analysis`](robot:meta-cdc-ku "wikilink")\
` * `[`Comparing` `Boone` `mega-experiment` `with` `our`
`screens`](robot:Boone-Lydall "wikilink")\
` * `[`Comparing` `our` `SGA` `experiment` `with` `some` `simulations`
`from` `the` `Manchester`
`CISB`](yeast2:ManchesterValidation "wikilink")

##### Other relevant wiki links

` * `[`robot` `working` `group`](robot:robot "wikilink")\
` * `[`HTP` `data` `meetings`](robot:high_throughput_data "wikilink")\
` * `[`heterogeneity`](hetero:hetero "wikilink")\
` * `[`biomath` `working` `group`](biomath "wikilink")\
` * `[`yeast` `group`](yeast:yeasthome "wikilink")\
` * `[`insilico` `group`](insilico:insilicohome "wikilink")

##### Relevant external links

` * Lots of useful yeast info from the  `[`SGD`](http://www.yeastgenome.org/ "wikilink")\
` * Also see the wikipedia page for `[`Saccharomyces`
`cerevisiae`](http://en.wikipedia.org/wiki/Saccharomyces_cerevisiae "wikilink")\
` * `[`BioGrid`](http://www.thebiogrid.org/ "wikilink")` may also be useful`

##### Statistical issues todo list

` * lighting correction`\
` * think carefully about **scaling the data** - sqrt(area) vs log(greyscale) etc. - what is the "best" measure to use?`\
` * random effects modelling`\
` * Bayesian modelling`

--- //[Darren Wilkinson](people:darren "wikilink") 2008/10/09 14:01//
