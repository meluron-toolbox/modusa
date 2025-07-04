Engine Creation Guide
=====================

In Modusa, all core logic **must reside in a separate `ModusaEngine` subclass**, not directly inside the plugin.

This ensures:

- Cleaner separation of concerns.
- Reusability of processing logic across plugins, tests, scripts, and CLI tools.
- Easier unit testing (engines don’t depend on plugin infrastructure).
- Plugins remain lightweight and act only as adapters.

Engine Structure
----------------

An **ModusaEngine** is a plain Python class or function that performs a specific computation or transformation. It should be completely **decoupled from the modusa signal and plugin framework**.

.. code-block::

    modusa/
        engines/
            my_engine.py


Basic Engine Example
--------------------

Let’s say we want to build a signal amplifier. The engine looks like:

.. code-block:: python

    # modusa/engines/amplifier.py

    from modusa.signals import Signal1D

    class AmplifierEngine:
        def __init__(self, gain: float = 1.0):
            self.gain = gain

        def run(self, signal: Signal1D) -> Signal1D:
            y = signal.y * self.gain
            return Signal1D(y=y, x=signal.x)

Key Points:

- `run()` performs the transformation.
- Engine should take clean inputs and return clean outputs.
- Avoid direct dependency on plugins or decorators.


Connecting Engine to Plugin
---------------------------

Now that you have the engine, use it inside your plugin’s `apply()` method.

.. code-block:: python

    from modusa.plugins import ModusaPlugin
    from modusa.decorators import plugin_safety_check
    from modusa.signals import Signal1D
    from modusa.engines.amplifier import AmplifierEngine

    class AmplifyPlugin(ModusaPlugin):
        @property
        def allowed_input_signal_types(self):
            return (Signal1D, )

        @property
        def allowed_output_signal_types(self):
            return (Signal1D, )

        @plugin_safety_check()
        def apply(self, signal: Signal1D) -> Signal1D:
            engine = AmplifierEngine(gain=2.0)
            return engine.run(signal)

✅ This makes the plugin reusable and logic-free. Any CLI or GUI can now directly use the `AmplifierEngine` too.

Testing Engines
---------------

You can now unit test the engine without touching plugins:

.. code-block:: python

    from modusa.engines.amplifier import AmplifierEngine
    from modusa.signals import Signal1D
    import numpy as np

    def test_amplifier():
        signal = Signal1D(y=np.array([1, 2, 3]), x=np.array([0, 1, 2]))
        result = AmplifierEngine(gain=3).run(signal)
        assert all(result.y == np.array([3, 6, 9]))

This makes testing fast, independent, and framework-free.

Summary
-------

- ✅ Create all core logic in `modusa.engines`.
- ✅ Keep plugins thin and delegate work to engines.
- ✅ Test engines directly.
- ❌ Do **not** put complex logic directly in plugin `apply()` methods.

This architecture promotes modularity, testability, and maintainability across the Modusa ecosystem.
