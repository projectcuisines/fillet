## Scripts and Data to Generate Model Intercomparison Figures

Each model has its own directory that should contain subdirectories for each benchmark and experiment. To be compliant with the Protocol, each directory must adhere to the following structure:

```
CodeName/
├── ben1/
│   ├── case_0/
│   │   └─ lat_output.dat
│   └── global_output.dat
├── ben2/
│   ├── case_0/
│   │   └─ lat_output.dat
│   └── global_output.dat
├── ben3/
│   ├── case_0/
│   │   └─ lat_output.dat
│   └── global_output.dat
├── exp1/
│   └── global_output.dat
├── exp1a/
│   └── global_output.dat
├── exp2/
│   └── global_output.dat
├── exp2a/
│   └── global_output.dat
├── exp3_cold
│   └── global_output.dat
├── exp3_warm
│   └── global_output.dat
├── exp4_cold
│   └── global_output.dat
├── exp4_warm
│   └── global_output.dat
```

The structure of each `global_output.dat` and `lat_output.dat` must follow the formatting described in the [Templates](../Templates) directory.
Please do not add extra files to your directory as they are unnecessary and increase the size of the repository. 

To finish the process, add the new model directory and its name to the `model_dirs` and `labels` arrays at the top of each src/plot_*.py script.
