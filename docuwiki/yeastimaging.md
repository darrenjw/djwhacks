###### Yeast Live Cell Imaging

Project to look at live cell imaging of telomere and other DNA damage
responses using fluorescent reporters in yeast

##### General notes

James thinks it should be possible to image GFP and RFP on Dave's
microscope, but that CFP and YFP are likely to be more difficult. Lots
of fluorescent strains in the strain collection, including all the
strains from Lisby et al (2004).

Make slides as described in my [bacillus lab
notes](hetero:bacilluslabnotes#slides "wikilink"). Will put yeast
[microscopy info](hetero:yeastlabnotes#microscopy "wikilink") in my
[yeast lab notes](hetero:yeastlabnotes "wikilink"). Other stuff from my
[yeast front page](hetero:yeast "wikilink"). For deconvolution of
Z-stacks, see Glyn's page on [Huygens](resources:huygens "wikilink").

Lots of info on
[Wikipedia](http://en.wikipedia.org/wiki/Main_Page "wikilink"):
[Microscopy](http://en.wikipedia.org/wiki/Microscopy "wikilink"),
[Deconvolution](http://en.wikipedia.org/wiki/Deconvolution "wikilink"),
[Convolution](http://en.wikipedia.org/wiki/Convolution "wikilink"),
[Fluorescence
microscopy](http://en.wikipedia.org/wiki/Fluorescence_microscopy "wikilink"),
[Bright field
microscopy](http://en.wikipedia.org/wiki/Bright_field_microscopy "wikilink"),
[Light
microscope](http://en.wikipedia.org/wiki/Light_microscope "wikilink"),
[Numerical
aperture](http://en.wikipedia.org/wiki/Numerical_aperture "wikilink")
(NA), [Point spread
function](http://en.wikipedia.org/wiki/Point_spread_function "wikilink")
(PSF) ...

Also see the Nikon [microscopy
tutorials](http://www.microscopyu.com/index.html "wikilink")

Documents for Prior H122 z-motor which may be set up incorrectly,
causing problems with autofocus: , ,

##### Important strains

See the [strain
database](http://minch-moor.ncl.ac.uk/fmi/iwp/cgi?-db=FuriousFive&-startsession "wikilink")
for further details. All strains in the W303 background unless otherwise
stated.

\^Strain number\^Relevant genotype\^Comment| \^DLY640|MATa|W303
background "wild type" strain| \^DLY1109 |MATa
[cdc13](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=cdc13 "wikilink")-1|cdc13-1
reference strain| \^DLY3303 |MATa cdc13-1-int
[Exo1](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=exo1 "wikilink")-GFP::HIS3
| A GFP strain | \^DLY4211 |MATa
[bar1](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=bar1 "wikilink"):LEU2
[MRE11](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=mre11 "wikilink"):YFP
[RAD52](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=rad52 "wikilink")-RFP
| A strain containing YFP and RFP (from Lisby et al, 2004) | \^DLY4269
|MATa cdc13-1
[RAD9](http://db.yeastgenome.org/cgi-bin/locus.pl?locus=rad9 "wikilink"):GFP;HIS3MX6
| Another GFP strain |

##### Microscopy

#### Timelapse

` * **13/1/11** - `` growing on 50% YEPD, 1% agarose. Saved as AVI, but will load up in ImageJ as an image stack. Imaged using the 100x objective with oil. One frame every 2.5 minutes. Approx 5 hours in total, and was still growing at the end. Lack of autofocus a real problem, and slide was a bit of a botch, but shows that it is possible to do live cell imaging on this microscope.`\
` * **20/1/11** - Another `` - now growing on the proper media (used YEP rather than YEPD above), and on a decent slide. 2 hours, one frame every 2.5 minutes.`

#### Z-Stacks

` * **13/1/11** - `` - Z-stack for a single time-point. Brightfield only. Saved as AVI, but will load up in ImageJ as an image stack. Frames are 1um apart, going from -10um to 10um - 21 frames in total. Not yet sure what is possible to do with this, but hoping that Glyn can advise! ;-) First attempt at 3D deconvolution using ImageJ, giving a slice ``, and `` (slicing the top off the big cell). Interesting, but needs work...`\
` * **20/1/11** - Another `` Z-stack at a single time point. This time from 5um to -5um in slices of 0.2um - 51 frames in total. Another bash at 3d deconvolution. First a ``, and then the corresponding ``. `

##### Background and papers

` * `` - //Modelling the checkpoint response to telomere uncapping in`

budding yeast // - Paper describing Carole's model, and our
understanding of the telomere uncapping response more generally

` * `` - //Choreography of the DNA Damage Response:`

Spatiotemporal Relationships among Checkpoint and Repair Proteins// -
Live imaging of DNA damage with fluorescent reporters. We should have
these strains in the strain collection

` * `` - //Additional modules for versatile and economical PCR-based gene deletion and modification in S. Cerevisiae// - Techniques for gene knockouts, GFP tagging, etc.`\
` * `` - //TEN1 Is Essential for CDC13-Mediated Telomere Capping// - [James says: This is another paper looking at a temperature sensitive mutation in Ten1, believed to be part of the same complex as Cdc13 (i.e. Cdc13-Stn1-Ten1). Figure 7 might be the interesting one for you to look at. I don't know if this has specifically been done with cdc13-1, but I see no reason why In place of stn1-ts and fluorescent Rad52, you couldn't use cdc13-1 and fluorescent RPA (or Rad52).]`
