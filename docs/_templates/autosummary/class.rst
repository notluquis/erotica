{{ fullname | escape | underline }}

.. Do NOT remove the ``item != '__init__'`` filter below. It is correct, and it is correct only
   because ``docs/conf.py`` sets ``autoclass_content = "both"``: Sphinx then merges the
   ``__init__`` docstring into the class body, so listing ``__init__`` again under Methods would
   duplicate it. If that setting is ever reverted to ``"class"``, every constructor's Parameters
   block disappears from the built page entirely -- which is the state this repository was in
   until 2026-08-04.

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   {% for item in methods %}{% if item != '__init__' %}
      ~{{ name }}.{{ item }}
   {%- endif %}{% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
