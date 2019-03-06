# __init__.py

# Version of the ml_testing
__version__ = "0.1"

from . import columnar_tests
from . import classification_tests
from . import regression_tests
from . import structural_tests

__all__ = ["columnar_tests", "classification_tests",
           "regression_tests", "structural_tests", '__version__']

