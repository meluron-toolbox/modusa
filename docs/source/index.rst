modusa
======


**modusa** is a modular framework for signal processing and visualisation.

.. admonition:: **Motivation behind building 'modusa' project:**

   - We use numpy for numerical operations that are completely mathematical and index based (does not contain any semantics). The axis does not mean anything and every operation you do that changes the shape of the array, you need to recompute the new axis. This needs to be automated if you are dealing with physical signals that has meaning.
   - We use matplotlib for plotting which again is a general purpose library for visualisation. It will be much more user friendly if every physical signal automatically knows how to plot itself.
   - Same goes for scipy for signal based operations, all these libraries are built to provide us a general framefork.
   - 'modusa' library is trying to build on top of these libraries but with more meaningful semantic-rich operations for physical signals that you are working on and provding you the right tools for any physical signal based on the signal space it is currently in.
   - Another problem working with non-semantic libraries are that you can end up performing operations between two signals from different signal space as long as they are compatible. 'modusa' puts check in place to avoid such mistakes.

.. admonition:: **Who are the target users?**
   
   - **Researchers** working with digital signals.
   - **Educators** can use it for demonstrating signal processing concepts much easily without the hassle of bringing different tools from different places.
   - **Learners** can use it for experimenting and learning things hands-on.

.. toctree::
   :maxdepth: 1
   :caption: Quick Tour to Modusa
   
   quicktour/evolution
   Quick Guide <quicktour/qg>
   quicktour/contribution

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   
   tools/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   tutorials/index
   


   
   
   

