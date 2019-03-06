__version__ = '0.1'

from . import classification_tests
from . import columnar_tests
from . import regression_tests
from . import structural_tests

__all__ = ["classification_tests", "columnar_tests",
           "regression_tests", "structural_tests"]
