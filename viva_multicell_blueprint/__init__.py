"""viva_multicell_blueprint — workspace Python package."""

# Eagerly import the composites subpackage so that importing the workspace
# package (as the vivarium-workbench env-worker does via `import <package_path>`)
# fires the `@composite_generator` decorators and registers them for the live
# server. Without this, a top-level import leaves the generators unregistered
# and studies report "composite not found in registry".
from . import composites  # noqa: F401,E402
