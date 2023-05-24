# phoeniks

*This project is licensed under the terms of the GNU General Public License v3.0 or later.*

## ALPHA-RELEASE (Test at your own risk)

**phoeniks** stands for 

> **P**ULS **h**ands-**o**n **o**ptimized **e**xtraction **o**f $\textbf{n}-\textbf{i}\cdot \textbf{k’s}$

and aims to be an easy to use software, where a reference and sample trace from a THz-Time Domain Spectrometer (THz-TDS) can be inserted and refractive index and absorption coefficient of the sample under test can be extracted. It is free and open-source software (FOSS) and written in Python. It is focused on numerical extraction with minimal knowledge about the sample and (currently) supports one layer/interface.

It is developed by Tim Vogel, PhD student at the [Photonics and Ultrafast Laser Science (PULS)](https://www.puls.ruhr-uni-bochum.de/) at the Ruhr-University Bochum, Germany. 

It is not developed in a vacuum, but I want to credit many fruitful discussions with [Ioachim Pupeza](https://orcid.org/0000-0001-8422-667X), [Romain Peretti](https://orcid.org/0000-0002-1707-7341), [Andrew Burnett](https://orcid.org/0000-0003-2175-1893), Nicholas Greenall, and [Milan Öri](https://www.menlosystems.com/products/thz-time-domain-solutions/terak15-terahertz-spectrometer/) about THz-Time Domain Spectrometer, providing tips & tricks how to extract the refractive index and absorption coefficient.


There is no manual yet, but you can find this series of YouTube tutorials hopefully helpful:


[YouTube - Tutorial - How to download and install 00](https://www.youtube.com/watch?v=-7DV7OxYu_k&list=PLBl95THK44rPBusZgUx_J9c1wVAN4ob27&index=1)

<a href="https://www.youtube.com/watch?v=-7DV7OxYu_k&list=PLBl95THK44rPBusZgUx_J9c1wVAN4ob27&index=1" target="_blank">
 <img src="https://img.youtube.com/vi/-7DV7OxYu_k/hqdefault.jpg" alt="Phoeniks - Tutorial - How to download and install 00" border="10" />
</a>

[YouTube - Tutorial - Basic Extraction - Example 01](https://www.youtube.com/watch?v=QBSKeY-IRJc&list=PLBl95THK44rPBusZgUx_J9c1wVAN4ob27&index=2)

<a href="https://www.youtube.com/watch?v=QBSKeY-IRJc&list=PLBl95THK44rPBusZgUx_J9c1wVAN4ob27&index=2" target="_blank">
 <img src="https://img.youtube.com/vi/QBSKeY-IRJc/hqdefault.jpg" alt="Phoeniks - Tutorial - Basic Extraction - Example 01" border="10" />
</a>

---

## Credits:

This program is based, in its first version, on the Matlab code from Ioachim Pupeza.  Pupeza et al. introduced to limit the number of echos for a more realistic transfer-function as well as the spatially variant, moving average filter (SVMAF):

> Ioachim Pupeza, Rafal Wilk, Martin Koch.
> 
> „Highly Accurate Optical Material Parameter Determination with THz Time-Domain Spectroscopy“. 
> 
> Optics Express 15, Nr. 7 (2007): 4335. 
> 
> https://doi.org/10.1364/OE.15.004335

Part of the code were inspired by the excellent publications from the work group around Romain Peretti, which take a more qualitatively look on the inverse problem to extract the refractive index or, more general, the dielectric function of the sample under test:

> Romain Peretti, Sergey Mitryukovskiy, Kevin Froberger, Mohamed Aniss Mebarki, Sophie Eliet, Mathias Vanwolleghem, Jean-Francois Lampin. 
> 
> „THz-TDS Time-Trace Analysis for the Extraction of Material and Metamaterial Parameters“. 
> 
> IEEE Transactions on Terahertz Science and Technology 9, Nr. 2 (2019): 136–49.
> 
> https://doi.org/10.1109/TTHZ.2018.2889227

The group also developed a software with a graphical user interface called fit@TDS, which allows to explore different dielectric functions and how they fit to the measurement data:

https://github.com/THzbiophotonics/Fit-TDS

Nelly is another package, written in Matlab, which is focused to solve multi-layer problems, since it can create a transfer-function on-the-fly. It is developed in the Schmuttenmaer/Neu-group at the University of Yale:

> Uriel Tayvah, Jacob A. Spies, Jens Neu, Charles A. Schmuttenmaer.
> 
> „Nelly: A User-Friendly and Open-Source Implementation of Tree-Based Complex Refractive Index Analysis for Terahertz Spectroscopy“. 
> 
> Analytical Chemistry 93, Nr. 32 (2021): 11243–50. 
> 
> https://doi.org/10.1021/acs.analchem.1c02132

Nelly can be found under:

https://github.com/YaleTHz/nelly

The knowledge about refractive index extraction was also expanded by Nicholas Greenall, who wrote his thesis in the work group from Andrew Burnett.

> Nicholas Robert Greenall. 
> 
> „Parameter Extraction and Uncertainty in Terahertz Time-Domain Spectroscopic Measurements“.
> 
> (2017) PhD thesis, University of Leeds.
> 
> https://etheses.whiterose.ac.uk/19045/

Pixel art logo created at midjourney.com

TODO:
- [x]  Credit all people who helped to develop this program
- [x]  Pick a suitable open-source license
- [x]  Upload first program which runs
- [x]  Supply examples with artifical material
- [ ]  Supply examples with real measurements
- [ ]  Supply a manual/handbook, explaining the code (why is it doing what)
