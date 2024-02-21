#############################################

Description:

This dataset contains recordings of 32 sound producing insect species with a total 335 files and a length of 57 minutes. The dataset was compiled for training neural networks to automatically identify insect species while comparing adaptive, waveform-based frontends to conventional mel-spectrogram methods for audio feature extraction. This work will be submitted for publication in the future and the dataset can be used to replicate the results or for similar research. The scripts for audio processing and the machine learning implementations that were used will be published on Github (github.com/mariusfaiss/Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition).

Roughly half of the recordings (147) are of nine species belonging to the order Orthoptera. These recordings stem from a dataset that was originally compiled by Baudewijn Odé (unpublished). The remaining recordings (188) are of 23 species in the family Cicadidae. These recordings were selected from the Global Cicada Sound Collection hosted on Bioacoustica (doi.org/10.1093/database/bav054), including recordings published in doi.org/10.3897/BDJ.3.e5792 & doi.org/10.11646/zootaxa.4340.1. Many recordings from this collection included speech annotations in the beginning of the recordings, therefore the last ten seconds of audio were extracted and used in this dataset. 

All files were manually inspected and files with strong noise interference or with sounds of multiple species were removed. Between species, the number of files ranges from four to 22 files and the length from 40 seconds to almost nine minutes of audio material for a single species. The files range in length from less than one second to several minutes. All original files were available with sample rates of at least 44.1 kHz or higher but were resampled to 44.1 kHz mono WAV files for consistency.

#############################################

CSV:

The annotation files for both parts of the dataset include information for all recordings:

file_name:		The file name used in this dataset, with the species name attached
species:		The name of the species in the recording (see species list below)
class_ID:		A numeric class identifier relating to each species (0-31, see below)
data_set:		The data-subset the file was included in for training, testing or validating a neural network
original_file_name:	The original file name from the source dataset, without the species name

#############################################

Baudewijn Odé - Orthoptera:

Species				n	min
Chorthippus biguttulus		20	3:43
Chorthippus brunneus		13	2:15
Gryllus campestris		22	3:38
Nemobius sylvestris		18	8:54
Oecanthus pellucens		14	4:27
Pholidoptera griseoaptera	15	1:54
Pseudochorthippus parallelus	17	2:01
Roeseliana roeselii		12	1:03
Tettigonia viridissima		16	1:34

#############################################

Ed Baker - Cicadidae:
	
Species				n	min
Azanicada zuluensis		4	0:40
Brevisiana brevis		5	0:50
Kikihia muta			6	1:00
Myopsalta leona			7	1:10
Myopsalta longicauda		4	0:40
Myopsalta mackinlayi		7	1:08
Myopsalta melanobasis		5	0:43
Myopsalta xerograsidia		6	1:00
Platypleura capensis		6	1:00
Platypleura cfcatenata		22	3:34
Platypleura chalybaea		7	1:10
Platypleura deusta		9	1:23
Platypleura divisa		6	1:00
Platypleura haglundi		5	0:50
Platypleura hirtipennis		6	0:54
Platypleura intercapedinis	5	0:50
Platypleura plumosa		19	3:09
Platypleura sp04		8	1:20
Platypleura sp10		16	2:24
Platypleura sp11 cfhirtipennis	4	0:40
Platypleura sp12 cfhirtipennis	10	1:40
Platypleura sp13		12	2:00
Pycna semiclara			9	1:30

#############################################

class_ID list:

Azanicadazuluensis		0
Brevisianabrevis		1
Chorthippusbiguttulus		2
Chorthippusbrunneus		3
Grylluscampestris		4
Kikihiamuta			5
Myopsaltaleona			6
Myopsaltalongicauda		7
Myopsaltamackinlayi		8
Myopsaltamelanobasis		9
Myopsaltaxerograsidia		10
Nemobiussylvestris		11
Oecanthuspellucens		12
Pholidopteragriseoaptera	13
Platypleuracapensis		14
Platypleuracfcatenata		15
Platypleurachalybaea		16
Platypleuradeusta		17
Platypleuradivisa		18
Platypleurahaglundi		19
Platypleurahirtipennis		20
Platypleuraintercapedinis	21
Platypleuraplumosa		22
Platypleurasp04			23
Platypleurasp10			24
Platypleurasp11cfhirtipennis	25
Platypleurasp12cfhirtipennis	26
Platypleurasp13			27
Pseudochorthippusparallelus	28
Pycnasemiclara			29
Roeselianaroeselii		30
Tettigoniaviridissima		31

#############################################

Copyright (c) is held by the individual recordists.

These data are published under the Creative Commons Attribution licence (CC BY):
 https://creativecommons.org/licenses/by/4.0/
This licence allows to to re-use the data for almost any purpose (follow the link for more information), as long as you give credit to the original source.

For academic reuse, we ask that you do this as a citation to the research paper, given above.


