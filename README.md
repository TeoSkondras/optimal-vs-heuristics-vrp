# optimal-vs-heuristics-vrp

This repository compares optimal solvers vs heuristic algorithms for vehicle routing (CVRP and VRPTW). It contains analysis notebooks that load result CSVs, compute gaps to optimal, and produce summary plots.

## Contents
- compare_results_cvrp.ipynb — analysis notebook for CVRP results
- compare_results_vrptw.ipynb — analysis notebook for VRPTW (Solomon) results
- CVRP_Heuristic/results/ — heuristic result CSVs used by CVRP notebook
- CVRP_Optimal/Results/ — optimal solver CSVs for CVRP
- VRPTW_Heuristic/results/ — heuristic result CSVs used by VRPTW notebook
- VRPTW_Optimal/Results/ — optimal solver CSVs for VRPTW

## Expected input files (examples used by notebooks)
CVRP:
- CVRP_Heuristic/results/aug_setA_optimal_results.csv
- CVRP_Heuristic/results/CVRP_our_results.csv
- CVRP_Heuristic/results/CVRP_pyvrp_summary.csv
- CVRP_Optimal/Results/*.csv  (batch_results_*.csv files)

VRPTW:
- VRPTW_Heuristic/results/solomon_optimal_results.csv
- VRPTW_Heuristic/results/VRPTW_ours.csv
- VRPTW_Heuristic/results/vrptw_pyvrp_summary.csv
- VRPTW_Optimal/Results/*.csv  (batch_results_*.csv files)

If your file names differ, update the paths in the corresponding notebook cells.

## Dependencies
Install the required Python packages (example):
pip install pandas seaborn matplotlib jupyterlab

Optionally pin versions via a requirements.txt if needed.

## Running the analysis
1. Start Jupyter (e.g., jupyter lab) in the repository root:
   jupyter lab
2. Open one of the notebooks:
   - compare_results_cvrp.ipynb
   - compare_results_vrptw.ipynb
3. Run cells top-to-bottom. Notebooks assume the result CSVs are placed in the folders listed above.

## Notes
- Notebooks rename and normalize CSV columns (e.g., `runtime` → `time_to_best`, `objective` → `cost`) and create an `instance_name` column.
- The analysis computes percentage gap to optimal per instance and aggregates results by algorithm.
- Color palettes and algorithm ordering are defined inside the notebooks for plotting.