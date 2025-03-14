<p align="center">
  <img width = "225" src="docs/FILLET_logo.png?raw=true"/>
</p>

<h1 align="center">FILLET: Functionality of Ice Line Latitudinal EBM Tenacity<br>
(A <a href="https://nexss.info/cuisines/">CUISINES</a> model intercomparison project)<br>
  <a href="https://iopscience.iop.org/article/10.3847/PSJ/acba05">
    <img src="https://img.shields.io/badge/Read-Protocol_v1%2E0_paper-blue.svg?style=flat">
  </a>
</h2>


Welcome! You've found the hompage of FILLET, a comparison 
of one-dimensional energy balance models (EBMs) that have been applied to habitable exoplanets 
in peer-reviewed publications. The project began in 2022 as a part of CUISINES, with the first 
paper describing "Protocol v1.0" published the next year 
(<a href="https://iopscience.iop.org/article/10.3847/PSJ/acba05">Deitrick et al. 2023</a>).
It lays out the methodology for benchmarking code against the pre-industrial Earth,
as well as larger experiments that map out where in parameter space permanent ice sheets 
exist. Note that after the publication of Protocol v1.0, the initial intercomparisons revealed that
some parameters were meaningless for some implementations, so a revised Protocol, called v1.1, is
currently in preparation.

Results (in process) will consist of the actual comparisons between different community models. We're currently 
comparing the following codes: 
<a href="https://academic.oup.com/mnras/article/514/4/5105/6609498">ESTM</a>, 
<a href="https://iopscience.iop.org/article/10.3847/PSJ/ac49eb/pdf">HEXTOR</a>, 
<a href="https://iopscience.iop.org/article/10.3847/2041-8205/825/2/L21/pdf">Kadoya-Tajika</a>, 
<a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JE006160">OPS</a>,
<a href="https://iopscience.iop.org/article/10.3847/1538-4357/ab4da6/pdf">Shields-Bitz</a>, and 
<a href="https://github.com/VirtualPlanetaryLaboratory/vplanet">VPLanet-POISE</a>. If you have access to 
an EBM and would like to include it in future FILLET studies, please reach out to Chef 
[Rory Barnes](mailto:rory@astro.washington.edu).

This repository is organized as follows:
- Protocol1.0 contains scripts to generate the figures from the Protocol v1.0 paper
- Results contains scripts to compare EBMs following Protocol 1.0
- Templates contains instructions on how to pre-process your data for model intercomparison
- MeetingNotes contains the notes from ~monthly meetings
- src contains helpful scripts
