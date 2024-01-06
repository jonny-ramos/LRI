# LRI
Projects completed while working with the Sorg Lab at the Legacy Research Institute, Portland OR.

For the Novel Memory Reactivation Project:

Rats were trained to press a lever for an IV cocaine infusion reward on a fixed ratio 1 (FR1) schedule, that is every 1 press results in 1 reward. Rats were then given a microinjection into the medial prefrontal cortex (mPFC) of either chondroitinase ABC (ChABC) or Vehicle. After 72 hours rats were given a novel memory reactivation session where they were allowed to press freely for cocaine for 30 minutes on a variable ratio 5 schedule (VR5), that is rewards, are delivered after a random number of presses ranging from 1-9 (centered on 5). Local Field Potentials were recorded from the mPFC and dHPC during behavior sessions. Lever presses were timestamped and 2.5 second sections of data around each timestamp were epoched and taken as trials. Lever presses where a cocaine reward was delivered are referred to as "rewarded" trials and lever presses where no reward was delivered are referred to as "unrewarded". Animals who received a Vehicle microinjection are denoted as "Vehicle" and animals who received a ChABC microinjection are denoted as "ChABC". 

The code included here was used to manually curate trials via visual inspection and perform various LFP analyses. Not all analyses are accompanied by a plot and for these types of analyses, the output of analyzeLFP.py consists either of text describing statistical outcomes (notably for permutation testing) or an output binary file containing the resulting transformed data or computed metrics. Plotting for these analyses was carried in a separate series of jupyter notebooks, some of which are included here. In addition to the code and sample jupyter notebooks, I have included a pdf containing sample figures that were produced by the results of this code. 

References: 
PAC, mean vector length:
Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE, Berger MS, Barbaro NM, Knight RT. High gamma power is phase-locked to theta oscillations in human neocortex. Science. 2006 Sep 15;313(5793):1626-8. doi: 10.1126/science.1128115. PMID: 16973878; PMCID: PMC2628289.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2628289/ 

PAC, modulation index:
Tort AB, Kramer MA, Thorn C, Gibson DJ, Kubota Y, Graybiel AM, Kopell NJ. Dynamic cross-frequency couplings of local field potential oscillations in rat striatum and hippocampus during performance of a T-maze task. Proc Natl Acad Sci U S A. 2008 Dec 23;105(51):20517-22. doi: 10.1073/pnas.0810524105. Epub 2008 Dec 12. PMID: 19074268; PMCID: PMC2629291.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2629291/ 

Lag cross correlation:
Adhikari A, Sigurdsson T, Topiwala MA, Gordon JA. Cross-correlation of instantaneous amplitudes of field potential oscillations: a straightforward method to estimate the directionality and lag between brain areas. J Neurosci Methods. 2010 Aug 30;191(2):191-200. doi: 10.1016/j.jneumeth.2010.06.019. Epub 2010 Jun 30. PMID: 20600317; PMCID: PMC2924932.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2924932/ 

Granger prediction/causality
Seth AK, Chorley P, Barnett LC. Granger causality analysis of fMRI BOLD signals is invariant to hemodynamic convolution but not downsampling. Neuroimage. 2013 Jan 15;65:540-55. doi: 10.1016/j.neuroimage.2012.09.049. Epub 2012 Oct 2. PMID: 23036449. https://pubmed.ncbi.nlm.nih.gov/23036449/ 

Intertrial phase coherence
Florian Mormann, Klaus Lehnertz, Peter David, Christian E. Elger. Mean phase coherence as a measure for phase synchronization and its application to the EEG of epilepsy patients. Physica D: Nonlinear Phenomena, Volume 144, Issues 3–4, 2000, Pages 358-369. https://doi.org/10.1016/S0167-2789(00)00087-7.
https://www.sciencedirect.com/science/article/abs/pii/S0167278900000877 

Phase lag index
Stam CJ, Nolte G, Daffertshofer A. Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. Hum Brain Mapp. 2007 Nov;28(11):1178-93. doi: 10.1002/hbm.20346. PMID: 17266107; PMCID: PMC6871367. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6871367/ 
 
![image](https://github.com/jonny-ramos/LRI/assets/151964299/4a025e1d-0567-4b67-959b-a40aabefbdb4)
