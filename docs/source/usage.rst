Usage
=====

.. _installation:

Installation
------------

Neurosetta relies heavily on the Graph-tool and GUDHI libraries, both of which are python wrapped C++ libraries. Due to this, I would recommend setting up a virtual environment with these packages before installing Neurosetta from git.

Alternatively, there `requierments.yml` file in the root of the repository that can be used to set up a virtual environment with all the required packages.

You can use this file as follows to set up a conda virtual environment:

.. code-block:: console

    (base) conda env create -f requirements.yml


