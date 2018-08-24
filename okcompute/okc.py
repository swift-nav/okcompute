from functools import reduce
from inspect import signature, Parameter
import operator

import networkx as nx

from typing import NamedTuple, List, Any, Callable

import traceback

import okcompute
import datetime
import time

# TODO: Better hashing of fields / metrics (avoid collisions based on string names)
# TODO: Maybe add way of specifying a list fields with unknown name (thread names) with sub keys
# TODO: Make helper functions to reduce boiler plate in saving / resuming from intermediary processing
# TODO: Make reports returned by prune functions more consistent
# TODO: Standardize config/input/output conventions
# TODO: Should allow metric input/output be dicts?
# TODO: Python2 support?
# TODO: Function to generate input/metric descriptions


VALID_INPUT_PARAM = 'valid_input'

FIELD_DIVIDER = '/'
COL_DIVIDER = '+'


def DUMMY_VALIDATE(x): return True  # pylint: disable=unused-argument


def has_common(a, b):
    return bool(set(a) & set(b))


class Field(NamedTuple):
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    key: List[str]
    description: str
    validate_func: Callable[[Any], bool] = DUMMY_VALIDATE

    def key_to_str(self) -> str:
        return FIELD_DIVIDER.join(self.key)

    @staticmethod
    def str_to_key(str_key):
        return Field(str_key.split(FIELD_DIVIDER), '')

    def __hash__(self):
        return hash(self.key_to_str())

    def path_exists(self, root):
        """Try to access a nested object in root by item sequence."""
        try:
            self.get_by_path(root)
            return True
        except KeyError:
            return False

    def get_by_path(self, root):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, self.key, root)

    @staticmethod
    def get_field_set(root, fields):
        base_key = fields[0].key[:-1]
        entries = [field.key[-1] for field in fields]
        try:
            return reduce(operator.getitem, base_key, root)[entries]
        except TypeError:
            raise TypeError(
                f"object in {base_key} is not a DataFrame and can't return a column set")

    def set_by_path(self, root, value):
        """Set a value in a nested object in root by item sequence."""
        val = reduce(operator.getitem, self.key[:-1], root)
        val[self.key[-1]] = value

    def __eq__(self, other):
        return self.key_to_str() == other


class MetricSpec(NamedTuple):
    name: str
    description: str
    func: Callable
    input_fields: List[Field]
    output_fields: List[Field]
    has_fallback_output: bool = False
    optional_fields: List[Field] = []

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
        assert not self.G.has_node(spec), f'Duplicate metric {spec}'
        self.G.add_node(spec, color='green', use_input=True, default_fields=[])

        for input_field in spec.input_fields:
            if not isinstance(input_field, list):
                input_field = [input_field]
            for sub_field in input_field:
                self.G.add_node(sub_field, color='turquoise')
                self.G.add_edge(sub_field, spec)

        for output_field in spec.output_fields:
            assert not self.G.has_node(
                output_field), f'Duplicate output field {output_field}'
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
        if type(node) == MetricSpec:
            removed_metrics.add(node)
        self.G.remove_node(node)
        for child in children:
            if type(child) == MetricSpec:
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

    def prune_inputs(self, input_data, invalid_input=None):
        """Computes minimin graph required to compute targets from graph
        """
        desired_inputs = self.get_inputs()
        if invalid_input is None:
            invalid_input = {}
            for in_field in desired_inputs:
                if in_field.path_exists(input_data):
                    try:
                        if not in_field.validate_func(in_field.get_by_path(input_data)):
                            invalid_input[in_field] = "Validation failed"
                    except Exception:  # pylint: disable=broad-except
                        invalid_input[in_field] = f"Validation exception: {traceback.format_exc()}"
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

    def prune_precalculated_metrics(self, input_data):
        """Computes minimin graph required to compute targets from graph
        """
        skipped_metrics = set()
        skipped_edges = set()
        for node in self.G.nodes():
            if type(node) != MetricSpec:
                continue
            satisfied = True
            for d in self.G.successors(node):
                if not d.path_exists(input_data):
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

    def run(self, input_data):
        run_report = {}
        processed = []
        loop_needed = True
        missing = {}
        while loop_needed:
            loop_needed = False
            for node in nx.topological_sort(self.G):
                if type(node) != MetricSpec or node in processed:
                    continue
                start_time = time.perf_counter()
                result = 'Success'
                try:
                    if not self.G.nodes[node]['use_input']:
                        inputs = [None for arg in node.input_fields]
                        retvals = node.func(*inputs, valid_input=False)
                    else:
                        kwargs = {}
                        func_params = signature(node.func).parameters
                        for arg_name, field in zip(func_params, node.input_fields):
                            defaults = self.G.nodes[node]['default_fields']
                            if isinstance(field, list):
                                if not has_common(defaults, field):
                                    kwargs[arg_name] = Field.get_field_set(
                                        input_data, field)
                            else:
                                if field not in defaults:
                                    kwargs[arg_name] = field.get_by_path(
                                        input_data)
                        if node.has_fallback_output:
                            kwargs[VALID_INPUT_PARAM] = True
                        retvals = node.func(**kwargs)
                    if not isinstance(retvals, tuple) or (len(node.output_fields) == 1 and len(retvals) != 1):
                        retvals = (retvals,)
                    assert len(node.output_fields) == len(
                        retvals), "Metric didn't produce expected number of outputs"
                    for field, val in zip(node.output_fields, retvals):
                        field.set_by_path(input_data, val)
                except Exception:  # pylint: disable=broad-except
                    result = 'Failure: ' + traceback.format_exc()
                stop_time = time.perf_counter()
                run_report[node.name] = {
                    'elapsed': stop_time - start_time,
                    'result': result
                }
                processed.append(node)
                if result != 'Success':
                    loop_needed = True
                    invalid_inputs = {
                        field: f'Missing due to {node.name} failure' for field in self.G.successors(node)}
                    self.G.remove_node(node)
                    missing.update(self.prune_inputs(
                        input_data, invalid_inputs))
                    break
        return run_report, missing


    def save_graph(self, out_file):
        def mapping(node):
            if isinstance(node, MetricSpec):
                return node.name
            if isinstance(node, Field):
                return node.key_to_str()
        A=nx.nx_agraph.to_agraph(nx.relabel_nodes(self.G,mapping))
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
        field_nodes = [node for node in self.G.nodes() if isinstance(node, Field)]
        return field_nodes

    def get_metrics(self):
        metric_nodes = [node for node in self.G.nodes() if isinstance(node, MetricSpec)]
        return metric_nodes


class App:
    def __init__(self, name: str, description: str, version: str) -> None:
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
            output_fields) > 0, f"Metric output_fields must have at least one entry"
        assert arg_count == len(
            input_fields), f"Metric input_fields count:{len(input_fields)} doesn't match function input count:{arg_count} (ignoring valid_input)"
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
                    assert sub_field.key[:-1] == base_key, f'All sub fields in {field} must have same base object'
        spec = MetricSpec(name, description, func, input_fields,
                          output_fields, has_fallback_output, optional_fields)
        self.graph.add_node(spec)

    def metric(self, description, input_fields, output_fields):
        def deco(func):
            self.add_metric(func, description, input_fields, output_fields)
            return func
        return deco

    def save_graph(self, file_path):
        self.graph.save_graph(file_path)

    def run(self, input_data, desired_output_fields=None, dry_run=False, skip_existing_results=False, save_graph_path=None, meta_args=''):
        start_time = time.perf_counter()
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
            skipped_metrics = graph.prune_precalculated_metrics(input_data)
            report['existing_results_skipped'] = [
                metric.name for metric in skipped_metrics]
        report['unneeded_metrics'] = []
        if desired_output_fields is not None:
            skipped_metrics = graph.prune_outputs(desired_output_fields)
            report['unneeded_metrics'] = [
                metric.name for metric in skipped_metrics if type(metric) == MetricSpec]
        unavailable_metrics = graph.prune_inputs(input_data)
        report['metrics_missing_input'] = unavailable_metrics
        if not dry_run:
            report['run_results'], unavailable_metrics = graph.run(input_data)
            report['metrics_missing_input'].update(unavailable_metrics)
        stop_time = time.perf_counter()
        report['meta_data']['elapsed'] = stop_time - start_time

        if save_graph_path:
            graph.save_graph(save_graph_path)
        return report
