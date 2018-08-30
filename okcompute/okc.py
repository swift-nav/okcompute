# Copyright (C) 2018 Swift Navigation Inc.
# Contact: Swift Navigation <dev@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
"""
    okcompute.okc
    ~~~~~~~~~

    This modeule implements the App and Field classes used to create test sets
"""

import okcompute

import networkx as nx

import operator
import sys
import traceback
import datetime
import time
if sys.version_info > (3, 0):
    from inspect import signature, Parameter  # pylint: disable=no-name-in-module, import-error
    from functools import reduce  # pylint: disable=redefined-builtin
else:
    from funcsigs import signature, Parameter  # pylint: disable=no-name-in-module, import-error

VALID_INPUT_PARAM = 'valid_input'

FIELD_DIVIDER = '/'


def DUMMY_VALIDATE(x): return True  # pylint: disable=unused-argument


def has_common(a, b):
    return bool(set(a) & set(b))


class Field:
    """A pointer to data involved with analysis.

    Fields are used by metrics to specify the inputs and outputs. They countain
    a key, description, and optional validation.

    The key ['a', 'b', 'c'] would refer to the value in root['a']['b']['c']
    and is represented as "a/b/c".

    Attributes:
        key (List[str]): The hierarchal location of a data field.

        description (str): Description of the field.

        validate_func (Callable[[Any], bool]): A function for validating input
            data. "def DUMMY_VALIDATE(x): return True" by default.
    """

    def __init__(self, key, description,
                 validate_func=DUMMY_VALIDATE):
        self.key = key
        self.description = description
        self.validate_func = validate_func

    def key_to_str(self):
        """Return the str representation of the key.

        Returns:
            str: str representation of key ie. "a/b/c"
        """
        return FIELD_DIVIDER.join(self.key)

    def __hash__(self):
        return hash(self.key_to_str())

    def path_exists(self, root):
        """Try to access a nested object in root by item sequence.

        Args:
            root (dict): The data_map field refers to.

        Returns:
            bool: True if value referenced by self.key exists, False otherwise.
        """
        try:
            self.get_by_path(root)
            return True
        except KeyError:
            return False

    def get_by_path(self, root):
        """Access the nested object in root at self.key

        Args:
            root (dict): The data_map field refers to.

        Returns:
            object: The value at key.self

        Raises:
            KeyError: If the path for self.key isn't in root
        """
        return reduce(operator.getitem, self.key, root)

    @staticmethod
    def get_field_set(root, fields):
        """Access a Pandas Dataframe like object with mapping to the set of fields

        For example if:

        root = {'input':{'a':pandas.Dataframe({'foo':[],'bar':[],'bat':[]})}}

        fields = [Field(key=['input', 'a', 'foo']), Field(key=['input', 'a', 'bat'])]

        Then get_field_set would return root['input']['a'][['foo', 'bat']]

        Args:
            root (dict): The data_map fields refers to.

            fields (List[Field]): a list of fields that share a common path
                except for the last string. These different strings refer to
                columns in the the object at the common path.

        Returns:
            object: The referenced columns of the object at the common path

        Raises:
            KeyError: If the path for a key in fields isn't in root

            TypeError: The object shared by the fields cannot take a
                __getitem__ key that's a list of strings
        """
        base_key = fields[0].key[:-1]
        entries = [field.key[-1] for field in fields]
        try:
            return reduce(operator.getitem, base_key, root)[entries]
        except TypeError:
            raise TypeError(
                "object in {} is not a DataFrame and can't return a column set".format(base_key))

    def set_by_path(self, root, value):
        """Set a value in a nested object in root at self.key

        Args:
            root (dict): The data_map field refers to.
            value (any): value to set the item to

        Raises:
            KeyError: If the path up to self.key isn't in root
        """
        val = reduce(operator.getitem, self.key[:-1], root)
        val[self.key[-1]] = value

    def __eq__(self, other):
        return self.key_to_str() == other


class MetricSpec:
    def __init__(self, name, description, func, input_fields, output_fields,
                 has_fallback_output=False, optional_fields=None):
        self.name = name
        self.description = description
        self.func = func
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.has_fallback_output = has_fallback_output
        if optional_fields is None:
            self.optional_fields = []
        else:
            self.optional_fields = optional_fields

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other


class Graph:
    """Represents a computation graph
    """

    def __init__(self):
        self.G = nx.DiGraph()

    def copy(self):
        ret = Graph()
        ret.G = self.G.copy()
        return ret

    def clear_props(self):
        for node in self.G.nodes:
            self.G.nodes[node]['use_input'] = True
            self.G.nodes[node]['default_fields'] = []

    def add_node(self, spec):
        assert not self.G.has_node(spec), 'Duplicate metric {}'.format(spec)
        self.G.add_node(spec, color='green', use_input=True, default_fields=[])

        for input_field in spec.input_fields:
            if not isinstance(input_field, list):
                input_field = [input_field]
            for sub_field in input_field:
                self.G.add_node(sub_field, color='turquoise')
                self.G.add_edge(sub_field, spec)

        for output_field in spec.output_fields:
            assert not self.G.has_node(
                output_field), 'Duplicate output field {}'.format(output_field)
            self.G.add_node(output_field, color='turquoise')
            self.G.add_edge(spec, output_field)

    def prune_outputs(self, targets):
        """Computes minimin graph required to compute targets from graph
        """
        seen = set()
        needed = []
        while targets:
            node = targets.pop()
            needed.append(node)
            for d in self.G.predecessors(node):
                if d not in seen:
                    seen.add(d)
                    targets.append(d)
        not_needed = [node for node in self.G.nodes() if node not in needed]

        for node in not_needed:
            self.G.remove_node(node)

        return not_needed

    def recursive_get_children(self, node):
        direct_children = [child for child in self.G.successors(node)]
        all_children = set(direct_children)
        for child in direct_children:
            all_children.update(self.recursive_get_children(child))
        return all_children

    def remove_with_children(self, node):
        removed_metrics = set()
        default_metrics = set()
        if not self.G.has_node(node):
            return removed_metrics, default_metrics
        children = [child for child in self.G.successors(node)]
        if isinstance(node, MetricSpec):
            removed_metrics.add(node)
        self.G.remove_node(node)
        for child in children:
            if isinstance(child, MetricSpec):
                if node in child.optional_fields:
                    self.G.nodes[child]['default_fields'].append(node)
                    continue
                elif child.has_fallback_output:
                    self.G.nodes[child]['use_input'] = False
                    default_metrics.add(child)
                    continue
            child_removed_metrics, child_default_metrics = self.remove_with_children(
                child)
            removed_metrics.update(child_removed_metrics)
            default_metrics.update(child_default_metrics)
        return removed_metrics, default_metrics

    def prune_inputs(self, data_map, invalid_input=None):
        """Computes minimin graph required to compute targets from graph
        """
        desired_inputs = self.get_inputs()
        if invalid_input is None:
            invalid_input = {}
            for in_field in desired_inputs:
                if in_field.path_exists(data_map):
                    try:
                        if not in_field.validate_func(in_field.get_by_path(data_map)):
                            invalid_input[in_field] = "Validation failed"
                    except Exception:  # pylint: disable=broad-except
                        invalid_input[in_field] = "Validation exception: {}".format(
                            traceback.format_exc())
                else:
                    invalid_input[in_field] = "Missing input"
        metrics_missing_input = {}
        for field in invalid_input.keys():
            child_removed_metrics, child_default_metrics = self.remove_with_children(
                field)

            def gen_report(metrics, has_default):
                return {
                    metric.name: {'bad_field': field.key_to_str(), 'reason': invalid_input[field], 'has_default': has_default} for metric in metrics
                }
            metrics_missing_input.update(
                gen_report(child_removed_metrics, False))
            metrics_missing_input.update(
                gen_report(child_default_metrics, True))
        return metrics_missing_input

    def prune_precalculated_metrics(self, data_map):
        """Computes minimin graph required to compute targets from graph
        """
        skipped_metrics = set()
        skipped_edges = set()
        for node in self.G.nodes():
            if not isinstance(node, MetricSpec):
                continue
            satisfied = True
            for d in self.G.successors(node):
                if not d.path_exists(data_map):
                    satisfied = False
                else:
                    skipped_edges.add((node, d))
            if satisfied:
                skipped_metrics.add(node)
        for edge in skipped_edges:
            self.G.remove_edge(*edge)
        for node in skipped_metrics:
            self.G.remove_node(node)
        return skipped_metrics

    def run(self, data_map):
        run_report = {}
        processed = []
        loop_needed = True
        missing = {}
        while loop_needed:
            loop_needed = False
            for node in nx.topological_sort(self.G):
                if not isinstance(node, MetricSpec) or node in processed:
                    continue
                start_time = time.clock()
                result = 'Success'
                try:
                    if not self.G.nodes[node]['use_input']:
                        inputs = [None] * len(node.input_fields)
                        retvals = node.func(*inputs, valid_input=False)
                    else:
                        kwargs = {}
                        func_params = signature(node.func).parameters
                        for arg_name, field in zip(func_params, node.input_fields):
                            defaults = self.G.nodes[node]['default_fields']
                            if isinstance(field, list):
                                if not has_common(defaults, field):
                                    kwargs[arg_name] = Field.get_field_set(
                                        data_map, field)
                            else:
                                if field not in defaults:
                                    kwargs[arg_name] = field.get_by_path(
                                        data_map)
                        if node.has_fallback_output:
                            kwargs[VALID_INPUT_PARAM] = True
                        retvals = node.func(**kwargs)
                    if not isinstance(retvals, tuple) or (len(node.output_fields) == 1 and len(retvals) != 1):
                        retvals = (retvals,)
                    assert len(node.output_fields) == len(
                        retvals), "Metric didn't produce expected number of outputs"
                    for field, val in zip(node.output_fields, retvals):
                        field.set_by_path(data_map, val)
                except Exception:  # pylint: disable=broad-except
                    result = 'Failure: ' + traceback.format_exc()
                stop_time = time.clock()
                run_report[node.name] = {
                    'elapsed': stop_time - start_time,
                    'result': result
                }
                processed.append(node)
                if result != 'Success':
                    loop_needed = True
                    invalid_inputs = {
                        field: 'Missing due to {} failure'.format(node.name) for field in self.G.successors(node)}
                    self.G.remove_node(node)
                    missing.update(self.prune_inputs(
                        data_map, invalid_inputs))
                    break
        return run_report, missing

    def save_graph(self, out_file):
        def mapping(node):
            if isinstance(node, MetricSpec):
                return node.name
            if isinstance(node, Field):
                return node.key_to_str()
        A = nx.nx_agraph.to_agraph(nx.relabel_nodes(self.G, mapping))
        A.draw(out_file, format='jpg', prog='dot')

    def get_inputs(self):
        in_nodes = [node for node in self.G.nodes(
        ) if self.G.in_degree(node) == 0]
        return in_nodes

    def get_internal(self):
        in_nodes = [node for node in self.G.nodes(
        ) if isinstance(node, Field) and self.G.out_degree(node) != 0 and self.G.in_degree(node) != 0]
        return in_nodes

    def get_outputs(self):
        out_nodes = [node for node in self.G.nodes(
        ) if self.G.out_degree(node) == 0]
        return out_nodes

    def get_fields(self):
        field_nodes = [node for node in self.G.nodes()
                       if isinstance(node, Field)]
        return field_nodes

    def get_metrics(self):
        metric_nodes = [node for node in self.G.nodes()
                        if isinstance(node, MetricSpec)]
        return metric_nodes


class App:
    """An app for performing a set of analyisis

    The metrics for analysis are specified by adding the metric decorator of
    an instance of this class to the anlysis functions.

    Specifying these metrics builds a dependancy graph for the analysis to
    perform. An image of this graph can be saved with save_graph.

    The analysis can be run on a data_map for the input and output with the
    run command. This command returns a report of what happened during the
    processing and the data_map is updated with results.

    Attributes:
        name (str): The name of the analysis set.
        description (str): A description for analysis set.
        version (str): A version string for analysis set.

    """

    def __init__(self, name, description, version):
        self.name = name
        self.description = description
        self.version = version
        self.graph = Graph()

    def add_metric(self, func, description, input_fields, output_fields):
        name = getattr(func, '__name__')
        func_params = signature(func).parameters
        arg_count = len(func_params)
        has_fallback_output = VALID_INPUT_PARAM in func_params
        optional_fields = []
        if has_fallback_output:
            arg_count -= 1
        assert len(
            output_fields) > 0, "Metric output_fields must have at least one entry"
        assert arg_count == len(
            input_fields), "Metric input_fields count:{} doesn't match function input count:{} (ignoring valid_input)".format(len(input_fields), arg_count)
        for i, param in enumerate(func_params.values()):
            if param.name == VALID_INPUT_PARAM:
                continue
            if param.default is not Parameter.empty:
                if isinstance(input_fields[i], list):
                    optional_fields += input_fields[i]
                else:
                    optional_fields.append(input_fields[i])
        for field in input_fields:
            if isinstance(field, list):
                base_key = field[0].key[:-1]
                for sub_field in field:
                    assert sub_field.key[:-
                                         1] == base_key, 'All sub fields in {} must have same base object'.format(field)
        spec = MetricSpec(name, description, func, input_fields,
                          output_fields, has_fallback_output, optional_fields)
        self.graph.add_node(spec)

    def metric(self, description, input_fields, output_fields):
        """Decorator for adding metrics to app

        The expected use is like:

        .. code-block:: py

            @example_app.metric(
                input_fields=[FIELD_INT2],
                output_fields=[FIELD_OUT3],
                description='example node4'
            )
            def metrics(in_arg, valid_input):
                ...
                return val

        The call signature of the function being added is inspected and used
        to inplicitly assess the desired behavior. Specifically the __name__
        attribute and the parameters. This may make using lambas of additional
        decorators more complicated.

        Here is what is implicitly checked:

        * The metric name - This is taken from the __name__ attribute. For a
            function this is it's name

        * The input parameters - the input_fields specified are matched in order
            to the positional arguments of the function. An assertion is raised
            if the number of parameters doesn't match the number of fields.

        * Parameter default values - if a parameter specifies a default value,
            the input field it corrasponds to is considered optional for this
            metric. This means the metric will still run even if the field is
            missing

        * Special valid_input parameter - if a parameter with the name
            valid_input is specified it is not mapped to the input fields. It
            instead is set to True if the input fields are valid, or False if
            they are not. Default parameters are considered valid. If a
            valid_input parameter exists it is expected that the metric will
            return some fallback output if valid_input is False.

        Similar to the input_fields, the output_fields map to the return values
        of the function. If multiple outputs are specified they are expected as
        a tuple with the same length as output_fields.

        Raises:

            AssertionError: If the call signature of the function doesn't match
                the input_fields, or the field are in some way invalid, this
                function will raise an assertion when the module is loaded.

        Args:
            input_fields (List[Field]): The input fields that map to the
                function parameters. As a special case, if one of the items in
                this list is a list itself, that set of fields is interpretted
                as columns for a Pandas Dataframe. See
                :func:`~okcompute.Field.get_field_set` for more details.

            output_fields (List[Field]): The function return values will be
                written to these fields

            description (str): A description for metric.

        Returns:
            callable: The decorated function

        """
        def deco(func):
            self.add_metric(func, description, input_fields, output_fields)
            return func
        return deco

    def save_graph(self, file_path):
        self.graph.save_graph(file_path)

    def run(self, data_map, desired_output_fields=None, dry_run=False,
            skip_existing_results=False, save_graph_path=None, meta_args=''):
        """Run the app's analysis using the data_map

        Any exceptions raised during a metric's function are surpressed. The
        tracebacks are logged in the report['run_results': {'result': str}].
        An assertion will be logged in this result if a metric doesn't return
        the number of results corrasponding to its output_fields.

        Args:
            data_map (dict): A dict that holds the inputs and outputs for
                metrics. The available inputs should be populated along with a
                dicts to countain internal and ouput fields

            desired_output_fields (List[Field]): A subset of the desired output
                fields. This will only run the metrics needed to produce these
                outputs. If this is None the metrics won't be skipped based on
                outputs.

            dry_run (bool): If this is true, don't actually run the analysis.
                Only produce a report checking the input for which metrics
                would be skipped.

            skip_existing_results (bool): If all the outputs for a metric are
                already preset in data_map, don't rerun the metric.

            save_graph_path (str): A path to save an image of the graph of
                analysis that runs based on the input. No graph is made if
                this path is None.

            meta_args (dict): Any additional values to add to the report.

        Returns:
            report (dict): A report of the analysis that was run. It countains
                the following top level fields:

                * meta_data - a description of the analysis and total run time

                * existing_results_skipped - if skip_existing_results is True
                    which metrics were skipped

                * unneeded_metrics - if desired_output_fields were specified
                    which metrics were skipped

                * metrics_missing_input - which metrics expected input missing
                    from data_map

                * run_results - elapsed time for metric and result (Success or
                    Failure along with cause)

        """

        start_time = time.clock()
        self.graph.clear_props()
        report = {
            'meta_data': {
                'okcompute_ver': okcompute.__version__,
                'run_time': str(datetime.datetime.now()),
                'app_meta': {
                    'app_name': self.name,
                    'app_version': self.version,
                },
                'meta_args': meta_args
            }
        }
        graph = self.graph.copy()
        report['existing_results_skipped'] = []
        if skip_existing_results:
            skipped_metrics = graph.prune_precalculated_metrics(data_map)
            report['existing_results_skipped'] = [
                metric.name for metric in skipped_metrics]
        report['unneeded_metrics'] = []
        if desired_output_fields is not None:
            skipped_metrics = graph.prune_outputs(desired_output_fields)
            report['unneeded_metrics'] = [
                metric.name for metric in skipped_metrics if isinstance(metric, MetricSpec)]
        unavailable_metrics = graph.prune_inputs(data_map)
        report['metrics_missing_input'] = unavailable_metrics
        if not dry_run:
            report['run_results'], unavailable_metrics = graph.run(data_map)
            report['metrics_missing_input'].update(unavailable_metrics)
        stop_time = time.clock()
        report['meta_data']['elapsed'] = stop_time - start_time

        if save_graph_path:
            graph.save_graph(save_graph_path)
        return report
