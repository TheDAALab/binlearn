There have been some refactoring in the codebase. I want you to go through the documentation in the docs folder, and update it accordingly. I also want you to add a couple of custom pages, explaining and illustrating some important design concepts:
* The support for numpy, pandas and polars arrays and dataframes, the logic behind preserve_dataframe, and the internal logic of how columns are represented across various ways of fitting and transforming.
* The fitted state reconstruction logic through get_params. Particularly, set_params(params) and constructor(**get_params) completely reconstructing a fitted state, while the params dict is always json serializable.

Also, update the README.rst accordingly!

I also want you to test all example codes, I want only working example codes in the documentation!
