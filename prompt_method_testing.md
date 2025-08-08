Write tests for the binning method to cover all lines. The tests should test the following scenarios:
* various types of input and output formats (numpy, pandas, polars), with preserve_dataframe True and False
* fitted state reconstruction via set_params(params) and constructor(**params), and testing if transform and inverse_transform work in the reconstructed state, without fit
* trying if fit works repeatedly, on the reconstructed state, too
* trying the various types of input and output formats after the fitted state is reconstructed through set_params(params) or constructor(**params)
* testing the sklearn pipeline integration
* edge cases with nans and infs and constant columns
