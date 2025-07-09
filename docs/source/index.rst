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
      
      - modusa **Signal**: Represents data with domain-specific tools (e.g., audio, music).
      - modusa **Plugin**: Connects engines to signals for execution.
      - modusa **Generator**: Creates new signals using rules or patterns.
      - modusa **IO**: Handles input/output, plotting, and playback.
      - modusa **Engine**: Core logic for processing and transforming signals.
      - modusa **Test**: Contains tests and fixtures.
      

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
   engines/index

#.. toctree::
#  :maxdepth: 1
#  :caption: How to contribute
#  
#  contrib/contribution_guidelines
#  contrib/engine_creation_guide
#  contrib/plugin_creation_guide
   
   
   

