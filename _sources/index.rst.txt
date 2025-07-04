modusa
======

**modusa** is a modular framework for audio signal analysis and processing, designed to help researchers and developers build reusable DSP chains with minimal code.

.. image:: images/core_components_nobg.png
   :alt: modusa components diagram
   :width: 100%
   :class: responsive-img
   :align: center
   
.. admonition:: **These are the 4 core components of modusa architecture**

      - modusa **Engine**: Contains the core logic for processing and transforming signals.
      - modusa **Plugin**: A thin layer that connects engines to signals for execution.
      - modusa **Signal**: Represents data with domain-specific tools (e.g., audio, music, etc).
      - modusa **Generator**: Creates new signals using predefined rules or patterns.

.. toctree::
   :maxdepth: 1
   :caption: Example Usage
   
   examples/Example1


.. toctree::
   :maxdepth: 1
   :caption: Public API
   
   signals
   generators
   plugins
   engines


.. toctree::
   :maxdepth: 1
   :caption: Developer Guide
   
   contribution_guidelines
   engine_creation_guide
   plugin_creation_guide
   
   
   

