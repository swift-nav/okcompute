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
from functools import partial


FIELD_INPUT_DUMMY1 = Field(
    ['input', 'dummy1'], 'Dummy input')
FIELD_INPUT_DUMMY2 = Field(
    ['input', 'dummy2'], 'Dummy input')
FIELD_INPUT_DUMMY3 = Field(
    ['input', 'dummy3'], 'Dummy input')
FIELD_INPUT_DUMMY4 = Field(
    ['input', 'dummy4'], 'Dummy input')
FIELD_INPUT_DUMMY_VALIDATION = Field(
    ['input', 'dummy_validation'], 'Dummy input', lambda x: x['foo'] is not None)

FIELD_RESULT_DUMMY_GOOD_NO_DEFAULT = Field(
    ['output', 'dummy_good_no_default'], 'Dummy analysis result')
FIELD_RESULT_DUMMY_GOOD_FALLBACK = Field(
    ['output', 'dummy_good_fallback'], 'Dummy analysis result')
FIELD_RESULT_DUMMY_GOOD_DEFAULT = Field(
    ['output', 'dummy_good_default'], 'Dummy analysis result')
FIELD_RESULT_DUMMY_VALIDATION = Field(
    ['output', 'dummy_validation'], 'Dummy analysis result')
FIELD_RESULT_DUMMY_BAD_OUT = Field(
    ['output', 'dummy_bad_out'], 'Dummy analysis result')
FIELD_RESULT_DUMMY_BAD_OUT2 = Field(
    ['output', 'dummy_bad_out2'], 'Dummy analysis result')
FIELD_RESULT_DUMMY_BAD_OUT3 = Field(
    ['output', 'dummy_bad_out3'], 'Dummy analysis result')

FIELDS_DUMMY_OUTPUT = [item for item in globals(
).keys() if item.startswith("FIELD_RESULT_")]

dummy_set = App('dummy_set', '', '1.0')


@dummy_set.metric(
    description='',
    input_fields=[FIELD_INPUT_DUMMY1],
    output_fields=[FIELD_RESULT_DUMMY_GOOD_NO_DEFAULT]
)
def dummy_good_no_default(dummy):
    return dummy


@dummy_set.metric(
    description='',
    input_fields=[FIELD_INPUT_DUMMY_VALIDATION],
    output_fields=[FIELD_RESULT_DUMMY_VALIDATION]
)
def dummy_validation(dummy):
    return dummy


@dummy_set.metric(
    description='',
    input_fields=[FIELD_INPUT_DUMMY1],
    output_fields=[FIELD_RESULT_DUMMY_GOOD_FALLBACK]
)
def dummy_good_fallback(dummy, valid_input):
    ret = "bar"
    if valid_input:
        ret = dummy
    return ret


@dummy_set.metric(
    description='',
    input_fields=[FIELD_INPUT_DUMMY1, FIELD_INPUT_DUMMY3, FIELD_INPUT_DUMMY2],
    output_fields=[FIELD_RESULT_DUMMY_GOOD_DEFAULT]
)
def dummy_good_default(dummy1, dummy3=None, dummy2=None):
    ret = dummy1
    if dummy3:
        ret += dummy3
    if dummy2:
        ret += dummy2
    return ret




def test_skip_all():
    report = dummy_set.run({}, desired_output_fields=[])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics())
    assert len(report['existing_results_skipped']) == 0
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 0


def test_good_one():
    data_map = {'input': {}, 'output': {}}
    val = 'foo'
    FIELD_INPUT_DUMMY1.set_by_path(data_map, val)
    report = dummy_set.run(data_map, desired_output_fields=[
                           FIELD_RESULT_DUMMY_GOOD_NO_DEFAULT])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 1
    assert report['run_results']['dummy_good_no_default']['result'] == "Success"
    assert FIELD_RESULT_DUMMY_GOOD_NO_DEFAULT.get_by_path(data_map) == val


def test_fallback():
    data_map = {'input': {}, 'output': {}}
    report = dummy_set.run(data_map, desired_output_fields=[
                           FIELD_RESULT_DUMMY_GOOD_FALLBACK])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert report['metrics_missing_input'] == {'dummy_good_fallback': {
        'bad_field': 'input/dummy1', 'has_default': True, 'reason': 'Missing input'}}
    assert len(report['run_results']) == 1
    assert report['run_results']['dummy_good_fallback']['result'] == "Success"
    assert FIELD_RESULT_DUMMY_GOOD_FALLBACK.get_by_path(data_map) == 'bar'
    val = 'foo'
    FIELD_INPUT_DUMMY1.set_by_path(data_map, val)
    report = dummy_set.run(data_map, desired_output_fields=[
                           FIELD_RESULT_DUMMY_GOOD_FALLBACK])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 1
    assert report['run_results']['dummy_good_fallback']['result'] == "Success"
    assert FIELD_RESULT_DUMMY_GOOD_FALLBACK.get_by_path(data_map) == val


def test_missing_one():
    data_map = {'input': {}, 'output': {}}
    report = dummy_set.run(data_map, desired_output_fields=[
                           FIELD_RESULT_DUMMY_GOOD_NO_DEFAULT])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert report['metrics_missing_input'] == {
        'dummy_good_no_default': {
            'bad_field': 'input/dummy1',
            'reason': 'Missing input',
            'has_default': False
        }
    }
    assert len(report['run_results']) == 0


def test_defaults():
    val1 = 'foo'
    val2 = 'bar'
    data_map = {'input': {}, 'output': {}}
    report = dummy_set.run(data_map, desired_output_fields=[FIELD_RESULT_DUMMY_GOOD_DEFAULT])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
      dummy_set.graph.get_metrics()) - 1
    assert len(report['existing_results_skipped']) == 0
    assert report['metrics_missing_input'] == {'dummy_good_default':
                                                 {'bad_field': 'input/dummy1',
                                                  'has_default': False,
                                                  'reason':'Missing input'
                                                  }
                                               }
    assert len(report['run_results']) == 0
    FIELD_INPUT_DUMMY1.set_by_path(data_map, val1)
    FIELD_INPUT_DUMMY2.set_by_path(data_map, val2)
    report = dummy_set.run(data_map, desired_output_fields=[FIELD_RESULT_DUMMY_GOOD_DEFAULT])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 1
    assert report['run_results']['dummy_good_default']['result'] == "Success"
    assert FIELD_RESULT_DUMMY_GOOD_DEFAULT.get_by_path(data_map) == val1+val2


def test_validation_good():
    data_map = {'input': {}, 'output': {}}
    val = {'foo': 'bar'}
    FIELD_INPUT_DUMMY_VALIDATION.set_by_path(data_map, val)
    report = dummy_set.run(data_map, desired_output_fields=[
                           FIELD_RESULT_DUMMY_VALIDATION])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 1
    assert report['run_results']['dummy_validation']['result'] == "Success"
    assert FIELD_RESULT_DUMMY_VALIDATION.get_by_path(data_map) == val


def test_validation_bad():
    data_map = {'input': {}, 'output': {}}
    val = {'foo': None}
    FIELD_INPUT_DUMMY_VALIDATION.set_by_path(data_map, val)
    report = dummy_set.run(data_map, desired_output_fields=[
                           FIELD_RESULT_DUMMY_VALIDATION])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert len(report['metrics_missing_input']) == 1
    assert report['metrics_missing_input'] == {
        'dummy_validation': {
            'bad_field': 'input/dummy_validation',
            'has_default': False,
            'reason': 'Validation failed'
        }
    }
    assert len(report['run_results']) == 0


def test_validation_exception():
    data_map = {'input': {}, 'output': {}}
    val = {'bar': 'bar'}
    FIELD_INPUT_DUMMY_VALIDATION.set_by_path(data_map, val)
    report = dummy_set.run(data_map, desired_output_fields=[
                           FIELD_RESULT_DUMMY_VALIDATION])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == len(
        dummy_set.graph.get_metrics()) - 1
    assert len(report['metrics_missing_input']) == 1
    assert report['metrics_missing_input']['dummy_validation']['reason'].startswith(
        "Validation exception")
    assert len(report['run_results']) == 0


def test_duplicate_metric():
    dummy_set2 = App('dummy_set', '', '1.0')

    @dummy_set2.metric(
        description='',
        input_fields=[FIELD_INPUT_DUMMY1],
        output_fields=[FIELD_RESULT_DUMMY_GOOD_NO_DEFAULT]
    )
    def dummy_good_no_default(dummy):
        return dummy
    with pytest.raises(AssertionError):
        @dummy_set2.metric(
            description='',
            input_fields=[FIELD_INPUT_DUMMY1],
            output_fields=[FIELD_RESULT_DUMMY_BAD_OUT]
        )
        def dummy_good_no_default(dummy):
            return dummy

    with pytest.raises(AssertionError):
        @dummy_set2.metric(
            description='',
            input_fields=[FIELD_INPUT_DUMMY1],
            output_fields=[FIELD_RESULT_DUMMY_GOOD_NO_DEFAULT]
        )
        def dummy_good_no_default2(dummy):
            return dummy


def test_input_mismatch():
    dummy_set2 = App('dummy_set', '', '1.0')
    with pytest.raises(AssertionError):
        @dummy_set2.metric(
            description='',
            input_fields=[FIELD_INPUT_DUMMY1],
            output_fields=[FIELD_RESULT_DUMMY_BAD_OUT]
        )
        def dummy_good_no_default1(dummy, foo):  # pylint: disable=unused-argument
            return
    with pytest.raises(AssertionError):
        @dummy_set2.metric(
            description='',
            input_fields=[FIELD_INPUT_DUMMY1],
            output_fields=[FIELD_RESULT_DUMMY_BAD_OUT]
        )
        def dummy_good_no_default2():
            return
    with pytest.raises(AssertionError):
        @dummy_set2.metric(
            description='',
            input_fields=[FIELD_INPUT_DUMMY1],
            output_fields=[FIELD_RESULT_DUMMY_BAD_OUT]
        )
        def dummy_good_no_default3(valid_input=True):  # pylint: disable=unused-argument
            return


def test_output_mismatch():
    def base_dummy(dummy):
        ret = 'foo'
        return tuple( ret for i in range(dummy) )
    dummy_set2 = App('dummy_set', '', '1.0')
    with pytest.raises(AssertionError):
        @dummy_set2.metric(
            description='',
            input_fields=[FIELD_INPUT_DUMMY1],
            output_fields=[]
        )
        def dummy_out0(dummy):
            return base_dummy(dummy)
    @dummy_set2.metric(
        description='',
        input_fields=[FIELD_INPUT_DUMMY1],
        output_fields=[FIELD_RESULT_DUMMY_BAD_OUT]
    )
    def dummy_out1(dummy):
        return base_dummy(dummy)
    @dummy_set2.metric(
        description='',
        input_fields=[FIELD_INPUT_DUMMY1],
        output_fields=[FIELD_RESULT_DUMMY_BAD_OUT2, FIELD_RESULT_DUMMY_BAD_OUT3]
    )
    def dummy_out2(dummy):
        return base_dummy(dummy)

    data_map = {'input': {}, 'output': {}}
    for out_count in range(4):
        FIELD_INPUT_DUMMY1.set_by_path(data_map, out_count)
        report = dummy_set2.run(data_map)
        assert len(report['existing_results_skipped']) == 0
        assert len(report['unneeded_metrics']) == 0
        assert len(report['metrics_missing_input']) == 0
        assert len(report['run_results']) == 2
        assert report['run_results']['dummy_out1']['result'] == "Success"
        if out_count == 2:
            assert report['run_results']['dummy_out2']['result'] == "Success"
        else:
            assert report['run_results']['dummy_out2']['result'].strip().endswith(
                "Metric didn't produce expected number of outputs")


IN1 = Field(
    ['input', 'in1'], 'Dummy input')
IN2 = Field(
    ['input', 'in2'], 'Dummy input')
IN3 = Field(
    ['input', 'in3'], 'Dummy input')
IN4 = Field(
    ['input', 'in4'], 'Dummy input')
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

class dummy_factory(object):
    def __init__(self):
        self.count = 0
        self.app = App('dummy_cascade_set', '', '1.0')
        self.fail_node_list = []

    def add_node(self, inputs, outputs, has_fallback=False):
        assert len(inputs) <= 10, 'Only supports generating node with less then 10 inputs'
        self.count += 1
        name = 'node' + str(self.count)
        def func_base(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, valid_input):
            if name in self.fail_node_list:
                raise ValueError
            if not valid_input:
                val = name + ' default'
            else:
                arg_list = [arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10]
                val = name + str([arg for arg in arg_list if arg is not None]).replace("'","")
            return tuple( val + str(i) for i in range(len(outputs)) )
        def func_no_default(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
            return func_base(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, True)
        func = func_base
        if not has_fallback:
            func = func_no_default
        args = [None] * (10 - len(inputs))
        partial_func = partial(func, *args)
        setattr(partial_func, '__name__', name)
        self.app.add_metric(partial_func, '', inputs, outputs)

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
    factory.add_node([IN1, IN2], [OUT1, INT1])
    factory.add_node([IN2], [INT2])
    factory.add_node([IN3, IN4], [INT3], True)
    factory.add_node([INT1, INT2], [OUT2])
    factory.add_node([INT2, INT3], [OUT3])
    return factory


def test_cascade(cascade_app_setup): # pylint: disable=redefined-outer-name
    data_map = {'input': {}, 'internal':{}, 'output': {}}
    app = cascade_app_setup.app
    del cascade_app_setup.fail_node_list[:]
    IN1.set_by_path(data_map, 'a')
    IN2.set_by_path(data_map, 'b')
    IN3.set_by_path(data_map, 'c')
    IN4.set_by_path(data_map, 'd')
    report = app.run(data_map, desired_output_fields=[
        OUT1,OUT2,OUT3])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert len(report['metrics_missing_input']) == 0
    assert len(report['run_results']) == 5
    for node in [ 'node' + str(i) for i in range(1, 6)]:
        assert report['run_results'][node]['result'] == 'Success'
    assert data_map['internal'] == {'int3': "node3[c, d]0",
                                      'int2': "node2[b]0",
                                      'int1': "node1[a, b]1"
                                     }
    assert data_map['output'] == {'out3': "node5[node2[b]0, node3[c, d]0]0",
                                    'out1': "node1[a, b]0",
                                    'out2': "node4[node1[a, b]1, node2[b]0]0"
                                   }


def test_cascade_error(cascade_app_setup): # pylint: disable=redefined-outer-name
    data_map = {'input': {}, 'internal':{}, 'output': {}}
    app = cascade_app_setup.app
    fail_nodes = cascade_app_setup.fail_node_list
    IN1.set_by_path(data_map, 'a')
    IN2.set_by_path(data_map, 'b')
    IN3.set_by_path(data_map, 'c')
    IN4.set_by_path(data_map, 'd')
    fail_nodes.append('node1')
    report = app.run(data_map, desired_output_fields=[
        OUT1,OUT2,OUT3])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert report['metrics_missing_input'] == {'node4':
        {'bad_field': 'internal/int1',
         'has_default': False,
         'reason': 'Missing due to node1 failure'
        }
    }
    assert len(report['run_results']) == 4
    for node in [ 'node' + str(i) for i in [2,3,5] ]:
        assert report['run_results'][node]['result'] == 'Success'
    assert data_map['internal'] == {'int3': "node3[c, d]0",
                                      'int2': "node2[b]0"
                                     }
    assert data_map['output'] == {'out3': "node5[node2[b]0, node3[c, d]0]0",
                                   }

def test_default_cascade(cascade_app_setup): # pylint: disable=redefined-outer-name
    data_map = {'input': {}, 'internal':{}, 'output': {}}
    app = cascade_app_setup.app
    del cascade_app_setup.fail_node_list[:]
    IN1.set_by_path(data_map, 'a')
    IN2.set_by_path(data_map, 'b')
    IN3.set_by_path(data_map, 'c')
    report = app.run(data_map, desired_output_fields=[
        OUT1,OUT2,OUT3])
    assert len(report['existing_results_skipped']) == 0
    assert len(report['unneeded_metrics']) == 0
    assert report['metrics_missing_input'] == {'node3':
        {'bad_field': 'input/in4',
         'has_default': True,
         'reason': 'Missing input'
        }
    }
    assert len(report['run_results']) == 5
    for node in [ 'node' + str(i) for i in range(1, 6)]:
        assert report['run_results'][node]['result'] == 'Success'
    assert data_map['internal'] == {'int3': "node3 default0",
                                      'int2': "node2[b]0",
                                      'int1': "node1[a, b]1"
                                     }
    assert data_map['output'] == {'out3': "node5[node2[b]0, node3 default0]0",
                                    'out1': "node1[a, b]0",
                                    'out2': "node4[node1[a, b]1, node2[b]0]0"
                                   }
