# OKCompute
A framework to make analysis stages clear, self documenting, and fault tolerant.

See [OKCompute Documentation](http://okcompute.swiftnav.com/) for details.

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
 * Better hashing of fields / metrics (avoid collisions based on string names)
 * Add way of specifying a list fields with name determined by data (thread names) with sub keys
 * Make helper functions to reduce boiler plate in saving / resuming from intermediary processing
 * Make reports returned by prune functions more consistent
 * Standardize config/input/output conventions
 * Should allow metric input/output be dicts?
