__version__ = '0.1'

from .classification_tests import classification_tests
from .columnar_tests import columnar_tests
from .regression_tests import regression_tests
from .structural_tests import structural_tests

__all__ = ["classification_tests", "columnar_tests", "regression_tests", "structural_tests"]
