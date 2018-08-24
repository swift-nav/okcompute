# -*- coding: utf-8 -*-
"""OKCompute Example Aplication

Provides command line interface to demenstrate basic framework usage
"""

import argparse
import json

from okcompute import Field, App

#: The okcompute app object. This is the handle for running analysis
example_app = App(name='example_app',
                  description='an example okcompute application',
                  version='1.0')

#: Fields used to reference data for analysis. The first parameter is the key.
# This key is a hierarchal path to a dict like object passed into the app to
# track state. The specific names (input/internal/output) are not special, and
# are just used here for clarity.

#: Input Fields. These map to data passed into the analysis
FIELD_IN1 = Field(key=['input', 'in1'],
                  description='dummy input 1')
FIELD_IN2 = Field(['input', 'in2'], 'dummy input 2')

#: Fields used to store intermediary values. These are values produced by
# one analysis node, to be used by others. These can also be passed in as
# input, and if skip_existing_results is used, the nodes that produce these
# ouptuts can be skipped
FIELD_INT1 = Field(['internal', 'int1'], 'dummy internal field 1')
FIELD_INT2 = Field(['internal', 'int2'], 'dummy internal field 2')

#: Fields used to store output
FIELD_OUT1 = Field(['output', 'out1'], 'dummy output 1')
FIELD_OUT2 = Field(['output', 'out2'], 'dummy output 2')
FIELD_OUT3 = Field(['output', 'out3'], 'dummy output 3')

nodes_to_fail = []

#  Example Structure
#  in1    in2
#    \   /  |
#   node1  node2
#    /  \     \
#  out1 int1  int2
#         \   / \
#         node3  node4
#           |      |
#         out2    out3

@example_app.metric(
    input_fields=[FIELD_IN1, FIELD_IN2],
    output_fields=[FIELD_OUT1, FIELD_INT1],
    description='example node1'
)
def node1(in1, in2='default1'):
    """Example analysis node

    This node needs FIELD_IN1 to run and FIELD_IN2 is optional
    It will generate output for FIELD_OUT1 and FIELD_INT1
    """
    if 'node1' in nodes_to_fail:
        raise AssertionError('Induced Failure')
    return f'node1_1[{in1}, {in2}]', f'node1_2[{in1}, {in2}]'


@example_app.metric(
    input_fields=[FIELD_IN2],
    output_fields=[FIELD_INT2],
    description='example node2'
)
def node2(in2):
    if 'node2' in nodes_to_fail:
        raise AssertionError('Induced Failure')
    return f'node2_1[{in2}]'


@example_app.metric(
    input_fields=[FIELD_INT1, FIELD_INT2],
    output_fields=[FIELD_OUT2],
    description='example node3'
)
def node3(int1, int2):
    if 'node3' in nodes_to_fail:
        raise AssertionError('Induced Failure')
    return f'node3_1[{int1}, {int2}]'


@example_app.metric(
    input_fields=[FIELD_INT2],
    output_fields=[FIELD_OUT3],
    description='example node4'
)
def node4(int2, valid_input):
    """Example analysis node

    This node needs FIELD_INT2 to run. If that field is missing, valid_input
    which is awill be False and node4 returns a fallback value for FIELD_OUT3.
    valid_input is a special parameter for metrics that is reserved for this.
    """
    if 'node4' in nodes_to_fail:
        raise AssertionError('Induced Failure')
    if valid_input:
        ret = f'node4_1[{int2}]'
    else:
        ret = f'node4_1[fallback]'
    return ret


def main():
    """
    Perform run example_app analysis based on command line arguments
    """
    parser = argparse.ArgumentParser(description='Run example_app')
    parser.add_argument('--in1', help='value for in1. Missing if not specified')
    parser.add_argument('--in2', help='value for in2. Missing if not specified')
    parser.add_argument('--fail-nodes', nargs='+',
                        help='induce failures in the specified nodes (node1-4)')
    parser.add_argument('--specify-outputs', nargs='+',
                        help='Only run analysis for specified outputs (out1-3)')
    parser.add_argument('--specify-internal', nargs='+',
                        help='Skip analysis by specifying intermediate values' \
                        ' (int1-2)')
    parser.add_argument('--save-graph', help='path to save graph of analysis')
    args = parser.parse_args()

    inputs = {}
    internal = {}
    desired_output_fields = None
    if args.in1:
        inputs['in1'] = args.in1
    if args.in2:
        inputs['in2'] = args.in2
    if args.fail_nodes:
        nodes_to_fail.extend(args.fail_nodes)
    if args.specify_internal:
        for val in args.specify_internal:
            internal[val] = f'{val}[Existing]'
    if args.specify_outputs:
        desired_output_fields = []
        output_mapping = {
            'out1': FIELD_OUT1,
            'out2': FIELD_OUT2,
            'out3': FIELD_OUT3,
        }
        for val in args.specify_outputs:
            desired_output_fields.append(output_mapping[val])
    data_map = {
        'input': inputs,
        'internal': internal,
        'output': {}
    }

    report = example_app.run(data_map,
                             desired_output_fields=desired_output_fields,
                             skip_existing_results=True,
                             save_graph_path=args.save_graph)

    print('Report:')
    print(json.dumps(report,
                     sort_keys=True,
                     indent=4,
                     separators=(',', ': ')))

    print('data_map:')
    print(json.dumps(data_map,
                     sort_keys=True,
                     indent=4,
                     separators=(',', ': ')))


if __name__ == '__main__':
    main()
