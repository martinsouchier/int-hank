# International HANK

This repository replicates the figures and tables in "[Exchange Rates and Monetary Policy with Heterogeneous Agents: Sizing up the Real Income Channel](http://web.stanford.edu/~aauclert/ha_oe.pdf)" (Auclert, Rognlie, Souchier and Straub 2024).

The code requires the [Sequence Space Jacobian (SSJ) toolkit](https://github.com/shade-econ/sequence-jacobian/) version 1.0, in addition to standard numerical Python packages (`numpy`, `scipy`, `matplotlib`, `numba`, `pandas`) and Jupyter notebooks. We have tested it using Python 3.12.4. The SSJ toolkit can be installed using `pip install sequence-jacobian`; please see the toolkit site for additional instructions.

## Organization

All of the results of the paper are obtained in main Jupyter notebook. Most of the code runs quickly, but the calibration takes time (a few minutes for the main model). You can select at the start of the notebook whether to calibrate the model or not.

There are supporting modules in three folders:

- `aux_model.py` specifies our various models .
- `aux_fn.py` has functions used to calibrate, to solve the model and to show results.
- `data/` includes the raw input data used in the calibration.
