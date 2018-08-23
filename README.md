# OKCompute
A framework to make analysis stages clear, self documenting, and fault tolerant.

## Key Features

 * Graph of dependencies - Can figure out minimum analysis for set of outputs, or diagnose missing inputs
 * Minimum Boilerplate
 * Human Readable Reports - Generates HTML documentation implicitly inferred from code and comprehensive reports of what occured during a run
 * Support for Pandas dataframes with column validation
 * Can specify optional fields or a fallback value if a required field is missing
 * Full stack traces are logged in the run results if an exception occurs during analysis
 * Supports checking for intermediary results to avoid rerunning slow analysis steps
 * Makes writing unit tests extremely easy

## TODO
 * Make generated documentation prettier
 * Document framework

