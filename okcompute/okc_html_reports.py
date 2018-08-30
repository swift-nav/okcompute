# Copyright (C) 2018 Swift Navigation Inc.
# Contact: Swift Navigation <dev@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

from jinja2 import Template  # pylint: disable=import-error
import inspect
from okcompute import okc


def generate_field_list(field_type, fields):
    field_template = '''<h1>{{field_type}}</h1><a id="{{field_type}}"></a>
    <ul>
        {% for field in fields %}
        <li><b>{{field.key_to_str()}}</b>: {{field.description}}<a id="{{field.key_to_str()}}"></a></li>
        {% if field.key_to_str() in validates %}
        Extra Validation:
        <pre><code>{{validates[field.key_to_str()]}}</code></pre>
        {% endif %}
        {% endfor %}
    </ul>'''
    validates = {}
    for field in fields:
        if field.validate_func != okc.DUMMY_VALIDATE:  # pylint: disable=bad-option-value, comparison-with-callable
            try:
                validates[field.key_to_str()] = inspect.getsource(field.validate_func)
            except TypeError:
                partial_func = field.validate_func
                validates[field.key_to_str()] = str(partial_func.keywords) + '\n' + inspect.getsource(partial_func.func)
    return Template(field_template).render(fields=fields, validates=validates, field_type=field_type)


def generate_metric_list(metrics):
    field_template = '''
    {% for metric in metrics %}
    <h3>{{metric.name}}</h3>
    <pre>{{metric.description}}</pre>
    <h4>Inputs</h4>
    <ul>
        {% for field in metric.input_fields %}
        {% if field in input_sets[metric] %}
        <li>
            Dataframe {{'/'.join(field[0].key_to_str().split('/')[:-1])}} with columns: [ {% for subfield in field %}<a href=#{{subfield.key_to_str()}}>{{subfield.key_to_str().split('/')[-1]}}</a>, {% endfor %} ]
        </li>
        {% else %}
        <li><a href=#{{field.key_to_str()}}><b>{{field.key_to_str()}}</b>: {{field.description}}</a></li>
        {% endif %}
        {% endfor %}
    </ul>
    <h4>Outputs</h4>
    <ul>
        {% for field in metric.output_fields %}
        <li><a href=#{{field.key_to_str()}}><b>{{field.key_to_str()}}</b>: {{field.description}}</a></li>
        {% endfor %}
    </ul>
    {% endfor %}
'''
    input_sets = { metric:[ field for field in metric.input_fields if isinstance(field, list) ] for metric in metrics }
    return Template(field_template).render(metrics=metrics, input_sets=input_sets)



template = '''<!DOCTYPE html>
<html>
  <head>
    <title>{{app_name}} Summary</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="http://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css" rel="stylesheet" media="screen">
    <style type="text/css">
      .container {
        max-width: 500px;
        padding-top: 100px;
      }
    </style>
  </head>
  <body>
    <img src="{{app_name}}.jpg" alt="{{app_name}} graph">
    <h1>Index</h1>
    {% for field in field_dict %}
    <h3><a href="#{{field}}">{{field}} Fields</a></h3>
    {% endfor %}
    <h3><a href="#Metrics">Metrics</a></h3>
    {% for field in field_dict %}
    {{field_dict[field]}}
    {% endfor %}
    <h1>Metrics</h1><a id="Metrics"></a>
    {{metrics}}
    <script src="http://code.jquery.com/jquery-1.10.2.min.js"></script>
    <script src="http://netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js"></script>
  </body>
</html>'''


def generate_app_doc(app, output_dir):
    field_dict = {
        'Input': generate_field_list('Input', app.graph.get_inputs()),
        'Output': generate_field_list('Output', app.graph.get_outputs()),
        'Internal': generate_field_list('Internal', app.graph.get_internal())
    }
    metrics = generate_metric_list(app.graph.get_metrics())
    with open('{}{}.html'.format(output_dir, app.name), 'w') as fd:
        fd.write(Template(template).render(app_name=app.name,
                                            field_dict=field_dict,
                                            metrics=metrics))
    app.save_graph('{}{}.jpg'.format(output_dir, app.name))
