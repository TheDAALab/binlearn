Design Concepts
===============

This section provides in-depth explanations of binlearn's core design concepts and architectural decisions. Understanding these concepts will help you make the most of binlearn's features and integrate it effectively into your data science workflows.

.. toctree::
   :maxdepth: 2
   :caption: Core Design Concepts

   dataframe_support
   fitted_state_reconstruction

DataFrame Support and Column Handling
------------------------------------

Learn how binlearn provides seamless support for numpy arrays, pandas DataFrames, and polars DataFrames while maintaining consistent behavior and column integrity across operations.

Key topics covered:

* **Multi-format input support** - Working with numpy, pandas, and polars
* **The preserve_dataframe parameter** - Controlling output formats  
* **Column representation system** - How columns are tracked internally
* **Guidance column separation** - Advanced column handling for supervised methods
* **Performance considerations** - Optimization tips for different formats

:doc:`Read more about DataFrame support <dataframe_support>`

Fitted State Reconstruction
--------------------------

Discover binlearn's powerful fitted state reconstruction system that enables complete model persistence and stateless transformations through JSON-serializable parameters.

Key topics covered:

* **get_params() and set_params()** - Complete state capture and restoration
* **JSON serialization support** - All parameters are JSON-serializable
* **Constructor reconstruction** - Alternative reconstruction methods
* **Pipeline integration** - Working with sklearn pipelines
* **Advanced scenarios** - Database storage, distributed computing, model serving

:doc:`Read more about fitted state reconstruction <fitted_state_reconstruction>`

Why These Concepts Matter
-------------------------

Understanding these design concepts helps you:

**Work More Effectively**
    Know when to use DataFrames vs arrays, how to optimize performance, and how to handle different data formats seamlessly.

**Build Better Pipelines**
    Leverage fitted state reconstruction for robust model persistence, A/B testing, and distributed processing.

**Integrate Successfully**
    Understand how binlearn integrates with pandas, polars, sklearn, and other data science tools.

**Debug Issues Faster**
    Understand the internal workings to quickly identify and resolve integration problems.

**Scale Applications**
    Use the design patterns effectively for production deployments and large-scale processing.

Common Integration Patterns
---------------------------

These concepts enable powerful integration patterns:

**Format-Agnostic Processing**
    Write code that works seamlessly with any supported data format:

    .. code-block:: python

        def universal_binning_function(data, n_bins=5):
            \"\"\"Works with numpy, pandas, or polars data.\"\"\"
            binner = EqualWidthBinning(n_bins=n_bins, preserve_dataframe=True)
            return binner.fit_transform(data)  # Output matches input format

**Stateless Model Serving**
    Deploy models without maintaining server state:

    .. code-block:: python

        def serve_binning_model(model_params_json, input_data):
            \"\"\"Stateless model serving endpoint.\"\"\"
            params = json.loads(model_params_json)
            binner = EqualWidthBinning(**params)  # Instantly ready
            return binner.transform(input_data)

**Pipeline Checkpointing**
    Save and restore complex processing pipelines:

    .. code-block:: python

        # Save pipeline state
        pipeline_state = {
            'step1_params': preprocessor.get_params(),
            'step2_params': binner.get_params(),
            'step3_params': postprocessor.get_params()
        }
        json.dump(pipeline_state, open('pipeline.json', 'w'))

**Cross-Framework Integration**
    Seamlessly integrate with pandas, polars, sklearn, and other tools while maintaining data format consistency and enabling model persistence.

Next Steps
----------

After reading these design concept guides, you'll be ready to:

* Use binlearn effectively in any data format environment
* Build robust, persistent model pipelines  
* Integrate binlearn into production systems
* Optimize performance for your specific use cases
* Troubleshoot integration issues quickly

Start with :doc:`dataframe_support` to understand how binlearn handles different data formats, then move on to :doc:`fitted_state_reconstruction` to learn about model persistence and reconstruction.
