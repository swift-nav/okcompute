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
    okcompute.report_utils
    ~~~~~~~~~~~~~~~~~~~~~~

    This modeule implements functions for displaying the report generated as
    the output of okc.App.run
"""

import sys
import json


def print_failures(report):
    """Print the analysis failures in a human readable format

        Args:
            report (dict): The dictionary output by okc.App.run

    """
    failures = 0
    for metric, result in report['run_results'].items():
        if result['result'] != 'Success':
            failures += 1
            print('======= Metric {} failed ======='.format(metric))
            print(result['result'].replace('\\n', '\n'))
    print('{} Failed Metrics Total'.format(failures))

if __name__ == '__main__':
    with open(sys.argv[1]) as fd:
        print_failures(json.load(fd))
