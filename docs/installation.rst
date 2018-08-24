.. _installation:

Installation
============

Python Version
--------------

We recommend using the latest version of Python 3. Python 2 is not currently
supported

Dependencies
------------

The core functionality of OKComputer requires

* `NetworkX`_ builds graphs for tracking dependancies

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

For certain features additional dependancies may be required. These can be
installed using the `Pip Extras`_ feature.

* `PyGraphviz`_ generates images based on dependancy graph
* `Jinja2`_ used to generate documentation based on the OKCompute application
* `MarkupSafe`_ Jinja2 dependancy

.. _Pip Extras: https://packaging.python.org/tutorials/installing-packages/#installing-setuptools-extras
.. _NetworkX: https://networkx.github.io/
.. _PyGraphviz: http://pygraphviz.github.io/
.. _Jinja2: http://jinja.pocoo.org/docs/
.. _MarkupSafe: https://www.palletsprojects.com/p/markupsafe/

Virtual environments
--------------------

Use a virtual environment to manage the dependencies for your project, both in
development and in production.

What problem does a virtual environment solve? The more Python projects you
have, the more likely it is that you need to work with different versions of
Python libraries, or even Python itself. Newer versions of libraries for one
project can break compatibility in another project.

Virtual environments are independent groups of Python libraries, one for each
project. Packages installed for one project will not affect other projects or
the operating system's packages.

Python 3 comes bundled with the :mod:`venv` module to create virtual
environments. If you're using a modern version of Python, you can continue on
to the next section.

.. _install-create-env:

Create an environment
~~~~~~~~~~~~~~~~~~~~~

Create a project folder and a :file:`venv` folder within:

.. code-block:: sh

    mkdir myproject
    cd myproject
    python3 -m venv venv

On Windows:

.. code-block:: bat

    py -3 -m venv venv

.. _install-activate-env:

Activate the environment
~~~~~~~~~~~~~~~~~~~~~~~~

Before you work on your project, activate the corresponding environment:

.. code-block:: sh

    . venv/bin/activate

On Windows:

.. code-block:: bat

    venv\Scripts\activate

Your shell prompt will change to show the name of the activated environment.

Install OKCompute
-------------

Within the activated environment, use the following command to install
OKCompute:

.. code-block:: sh

    pip install okcompute

To install the extras you can run one of the following commands:

.. code-block:: sh

    pip install okcompute[doc]
    pip install okcompute[plot]
    pip install okcompute[plot,doc]

Specifying "doc" lets you generate documentation from your application, and
"plot" is for generating images of the dependancy graphs

OKCompute is now installed. Check out the :doc:`/quickstart` or go to the
:doc:`Documentation Overview </index>`.

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

If you want to install the latest commit directly, you can run:

.. code-block:: sh

    pip install git+ssh://git@github.com/swift-nav/okcompute.git#egg=okcompute

If you have the code checked out locally you can install the pinned
dependancies with:

.. code-block:: sh

    pip install -r requirements.txt
    pip install -r requirements-extras.txt
    pip install -r requirements-test.txt

and create an `Editable Install`_ with

.. code-block:: sh

    pip install -e .

.. _Editable Install: https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs
