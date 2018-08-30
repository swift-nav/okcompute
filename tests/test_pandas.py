# Copyright (C) 2018 Swift Navigation Inc.
# Contact: Swift Navigation <dev@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

from okcompute import Field, App
import pytest
import pandas as pd

DUMMY_DATAFRAME = pd.DataFrame({'in1': [1, 2, 3],
                                'in2': [4, 5, 6],
                                'in3': [7, 8, 9],
                                'in4': [10, 11, 12]})

IN1 = Field(
    ['input', 'data', 'in1'], 'Dummy input')
IN2 = Field(
    ['input', 'data', 'in2'], 'Dummy input')
IN3 = Field(
    ['input', 'data', 'in3'], 'Dummy input')
IN4 = Field(
    ['input', 'data', 'in4'], 'Dummy input')
IN5 = Field(
    ['input', 'data', 'in5'], 'Dummy input')
IN_CONF = Field(
    ['input', 'config'], 'Dummy input')
INT1 = Field(
    ['internal', 'int1'], 'Dummy internal field')
INT2 = Field(
    ['internal', 'int2'], 'Dummy internal field')
INT3 = Field(
    ['internal', 'int3'], 'Dummy internal field')
OUT1 = Field(
    ['output', 'out1'], 'Dummy outout field')
OUT2 = Field(
    ['output', 'out2'], 'Dummy outout field')
OUT3 = Field(
    ['output', 'out3'], 'Dummy outout field')


def test_subfield_missmatch():
    with pytest.raises(AssertionError):
        dummy_set = App('dummy_set', '', '1.0')

        @dummy_set.metric(
            description='',
            input_fields=[[IN1, IN_CONF]],
            output_fields=[OUT1]
        ) # pylint: disable=unused-variable
        def test_subfield_missmatch1(df):
            return df

    dummy_set = App('dummy_set', '', '1.0')

    @dummy_set.metric(
        description='',
        input_fields=[[IN1, IN2]],
        output_fields=[OUT1]
    ) # pylint: disable=unused-variable
    def test_subfield_missmatch(df):
        return df
    data_map = {'input': {}, 'output': pd.DataFrame()}
    data_map['input']['data'] = {'in1': [1, 2, 3],
                                   'in2': [4, 5, 6]}
    report = dummy_set.run(data_map)
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert len(report['metrics_missing_input']) == 0
    assert report['run_results']['test_subfield_missmatch']['result'].strip().endswith(
        "is not a DataFrame and can't return a column set")


def test_mix():
    dummy_set = App('dummy_set', '', '1.0')
    val = 'foo'

    @dummy_set.metric(
        description='',
        input_fields=[[IN1, IN2], IN_CONF],
        output_fields=[OUT1]
    )
    def test_mix(df, conf):
        return pd.Series([conf + str(df.sum().sum())])

    data_map = {'input': {}, 'output': pd.DataFrame()}
    data_map['input']['data'] = DUMMY_DATAFRAME
    data_map['input']['config'] = 'foo'
    report = dummy_set.run(data_map)
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert len(report['metrics_missing_input']) == 0
    assert report['run_results']['test_mix']['result'] == "Success"
    test_sum = DUMMY_DATAFRAME[[IN1.key[-1], IN2.key[-1]]].sum().sum()
    assert OUT1.get_by_path(data_map)[0] == val + str(test_sum)


def test_missing_one():
    dummy_set = App('dummy_set', '', '1.0')

    @dummy_set.metric(
        description='',
        input_fields=[[IN1, IN5]],
        output_fields=[OUT1]
    )
    def test_missing_one(df):
        return pd.Series([df.sum().sum()])

    data_map = {'input': {}, 'output': pd.DataFrame()}
    data_map['input']['data'] = DUMMY_DATAFRAME
    report = dummy_set.run(data_map)
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert report['metrics_missing_input'] == {
        'test_missing_one': {
            'bad_field': 'input/data/in5',
            'reason': 'Missing input',
            'has_default': False
        }
    }
    assert len(report['run_results']) == 0


def test_defaults():
    dummy_set = App('dummy_set', '', '1.0')

    @dummy_set.metric(
        description='',
        input_fields=[IN1, [IN2, IN3]],
        output_fields=[OUT1]
    )
    def test_defaults(val, df=None):
        sums = val.sum()
        if df is not None:
            sums += df.sum().sum()
        return pd.Series([sums])

    data_map = {'input': {}, 'output': pd.DataFrame()}
    data_map['input']['data'] = DUMMY_DATAFRAME[[IN1.key[-1], IN2.key[-1]]]
    report = dummy_set.run(data_map)
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 1
    assert report['run_results']['test_defaults']['result'] == "Success"
    assert OUT1.get_by_path(data_map)[0] == 6
    data_map['input']['data'] = DUMMY_DATAFRAME
    report = dummy_set.run(data_map)
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 1
    assert report['run_results']['test_defaults']['result'] == "Success"
    assert OUT1.get_by_path(data_map)[0] == 45


class dummy_factory(object):
    def __init__(self):
        self.count = 0
        self.app = App('dummy_cascade_set', '', '1.0')
        self.fail_node_list = []

    def add_node(self, inputs, outputs, has_fallback=False):
        assert len(
            inputs) <= 10, 'Only supports generating node with less then 10 inputs'
        self.count += 1
        name = 'node' + str(self.count)

        def func_base(df, valid_input):
            if name in self.fail_node_list:
                raise ValueError
            if not valid_input:
                val = 1000
            else:
                val = df.sum().sum()
            return tuple(pd.Series(val * (i + 1)) for i in range(len(outputs)))

        def func_no_default(df):
            return func_base(df, True)

        func = func_base
        if not has_fallback:
            func = func_no_default
        setattr(func, '__name__', name)
        self.app.add_metric(func, '', inputs, outputs)


#      CASCADING GRAPH
#  in1    in2    in3    in4
#    \   / |        \    |
#   node1 node2      node3
#    /  \     |        |
#  out1 int1 int2    int3
#         \   / \    / \
#        node4   node5 node6
#          |      |     |
#        out2    out3  int4


@pytest.fixture(scope='module')
def cascade_app_setup():
    factory = dummy_factory()
    factory.add_node([[IN1, IN2]], [OUT1, INT1])
    factory.add_node([IN2], [INT2])
    factory.add_node([[IN3, IN4]], [INT3], True)
    factory.add_node([[INT1, INT2]], [OUT2])
    factory.add_node([[INT2, INT3]], [OUT3])
    return factory


def test_cascade(cascade_app_setup): # pylint: disable=redefined-outer-name
    data_map = {'input': {}, 'internal': pd.DataFrame(),
                  'output': pd.DataFrame()}
    app = cascade_app_setup.app
    del cascade_app_setup.fail_node_list[:]
    data_map['input']['data'] = DUMMY_DATAFRAME
    report = app.run(data_map, desired_output_fields=[
        OUT1, OUT2, OUT3])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 5
    for node in ['node' + str(i) for i in range(1, 6)]:
        assert report['run_results'][node]['result'] == 'Success'
    assert data_map['internal'].to_dict() == {'int3': {0: 57},
                                                'int2': {0: 15},
                                                'int1': {0: 42}
                                                }
    assert data_map['output'].to_dict() == {'out3': {0: 72},
                                              'out2': {0: 57},
                                              'out1': {0: 21}
                                              }


def test_cascade_error(cascade_app_setup): # pylint: disable=redefined-outer-name
    data_map = {'input': {}, 'internal': pd.DataFrame(),
                  'output': pd.DataFrame()}
    app = cascade_app_setup.app
    fail_nodes = cascade_app_setup.fail_node_list
    del fail_nodes[:]
    data_map['input']['data'] = DUMMY_DATAFRAME
    fail_nodes.append('node1')
    report = app.run(data_map, desired_output_fields=[
        OUT1, OUT2, OUT3])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert report['metrics_missing_input'] == {'node4':
                                               {'bad_field': 'internal/int1',
                                                'has_default': False,
                                                'reason': 'Missing due to node1 failure'
                                                }
                                               }
    assert len(report['run_results']) == 4
    for node in ['node' + str(i) for i in [2, 3, 5]]:
        assert report['run_results'][node]['result'] == 'Success'
    assert data_map['internal'].to_dict() == {'int3': {0: 57},
                                                'int2': {0: 15}
                                                }
    assert data_map['output'].to_dict() == {'out3': {0: 72},
                                              }


def test_default_cascade(cascade_app_setup): # pylint: disable=redefined-outer-name
    data_map = {'input': {}, 'internal': pd.DataFrame(),
                  'output': pd.DataFrame()}
    app = cascade_app_setup.app
    del cascade_app_setup.fail_node_list[:]
    data_map['input']['data'] = DUMMY_DATAFRAME[[
        IN1.key[-1], IN2.key[-1], IN3.key[-1]]]
    report = app.run(data_map, desired_output_fields=[
        OUT1, OUT2, OUT3])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert report['metrics_missing_input'] == {'node3':
                                               {'bad_field': 'input/data/in4',
                                                'has_default': True,
                                                'reason': 'Missing input'
                                                }
                                               }
    assert len(report['run_results']) == 5
    for node in ['node' + str(i) for i in range(1, 6)]:
        assert report['run_results'][node]['result'] == 'Success'
    assert data_map['internal'].to_dict() == {'int3': {0: 1000},
                                                'int2': {0: 15},
                                                'int1': {0: 42}
                                                }
    assert data_map['output'].to_dict() == {'out3': {0: 1015},
                                              'out1': {0: 21},
                                              'out2': {0: 57}
                                              }
