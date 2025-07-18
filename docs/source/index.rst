modusa
======

**modusa** is a modular framework for audio signal analysis and processing, designed to help researchers and developers build DSP chains with minimal code.

.. image:: images/core_components_nobg.png
   :target: index.html
   :alt: modusa components diagram
   :width: 60%
   :class: responsive-img
   :align: center
   
.. admonition:: **These are the 6 core components of modusa architecture**
      
      - **modusa Signal**: Core data representation with domain-specific utilities (e.g., audio/music signals).
      - **modusa Plugin**: Interfaces tools with signals, enabling chaining and execution.
      - **modusa Generator**: Synthesizes new signals using rules, templates, or patterns.
      - **modusa IO**: Manages input/output operations via loaders and savers.
      - **modusa Tools**: Utility classes and reusable components for signal processing workflows.
      - **modusa Test**: Contains tests and fixtures to ensure correctness and reliability.

.. toctree::
   :maxdepth: 1
   :caption: Quick Guide
   
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   signals/index
   plugins/index
   generators/index
   io/index
   tools/index

#.. toctree::
#  :maxdepth: 1
#  :caption: How to contribute
#  
#  contrib/contribution_guidelines
#  contrib/engine_creation_guide
#  contrib/plugin_creation_guide
   
   
   

