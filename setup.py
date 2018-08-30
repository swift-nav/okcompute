# Copyright (C) 2018 Swift Navigation Inc.
# Contact: Swift Navigation <dev@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.


import io
from setuptools import setup, find_packages

main_ns = {}
exec(open('okcompute/version.py').read(), main_ns)  # pylint: disable=exec-used

setup(
    name='okcompute',
    version=main_ns['__version__'],
    packages=find_packages(exclude=['tests*']),
    description=('Fault tolerant analysis framework.'),
    long_description=io.open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/swift-nav/okcompute',
    # use_scm_version=True,
    # setup_requires=['setuptools_scm'],
    install_requires=[
        'networkx',
    ],
    tests_require=[
        'pytest',
        'pandas',
    ],
    extras_require={
        'plot': ['pygraphviz'],
        'appdoc': ['pygraphviz', 'jinja2'],
        'doc': ['sphinx'],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
