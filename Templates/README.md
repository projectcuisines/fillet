
## Templates for output files 

All outputs should use plain text, space-separated columns based on the templates provided here. Add benchmark/experiment information to the headers. Use the `#` symbol for comments and notes. Add any notes regarding your simulation if you feel they are relevant--it doesn't hurt to have more information than necessary!

### Latitudinal outputs (see Table 5 in Protocol 1.0 paper) 

For each individual case (simulation), transcribe the latitudinal output into a file based on the template file `lat_output.dat`. This file will contain, as space-separated columns, latitude, surface temperature, surface albedo, top-of-atmosphere/planetary albedo (surface + atmosphere), and OLR. All should be annually-averaged over the final orbit after running each simulation to equilibrium.

For example:

```
# Use this header for notes and global details
# Name of benchmark/experiment: Benchmark 1
# Code: HEXTOR
# Case number: 0 
# Instellation (S_earth): 1
# XCO2 (ppm): 280
# Obliquity (degrees): 23.5

# Columns of data (annually averaged for last orbit)
# Lat = latitude (degrees)
# Tsurf = annually averaged surface temperature (K)
# Asurf = annually averaged surface albedo
# ATOA = annually averaged top-of-atmosphere/total/planetary albedo
# OLR = annually averaged outgoing longwave radiation (W m^-2)

# Lat Tsurf Asurf ATOA OLR
-88.0 240.19 0.6 0.7 134.30
-85.0 245.21 0.6 0.68 144.77
...
```

Note that the above data are fake, so don't attempt to replicate them!

### Global outputs (see Table 5 in Protocol 1.0 paper) 

For each benchmark or experiment, compile a list of the global properties using `global_output.dat` as a template. Give each individual simulation a case number (match this with the case number in the header in the latitudinal output file)--for the benchmarks there will be only 1 case. List as columns, the case number, instellation, obliquity, CO2 mixing ratio, global-annual mean surface temperature, and ice line latitude in the northern hemisphere and in the southern hemisphere. 

For example:
```
# Use this header for notes about experiment set up
# Compile all cases into a list

# Name of benchmark/experiment: Experiment 1
# Describe how ice line latitude is determined: where temperature falls below freezing in annual average
# Code: shields_bitz

# Columns of data
# Case = case number (0 -> total number of cases in experiment)
# Inst = instellation (S_earth; i.e., relative to Earth's 1361 W m^-2)
# Obl = obliquity (degrees)
# XCO2 = mixing ratio (volume) of CO2 (ppm)
# Tglob = global, annual mean surface temperature (K)
# IceLineN = latitude of ice line in northern hemisphere (deg)
# IceLineS = latitude of ice line in southern hemisphere (deg)

# Case Inst Obl XCO2 Tglob IceLineN IceLineS
0 0.85 0.0 280 230.05 0.0 0.0
1 0.95 0.0 280 277.7 66.0 65.99
...
```

Note that the above data are fake, so don't attempt to replicate them!
