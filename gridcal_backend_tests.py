import unittest
import warnings

# first the backend class (for the example here)
from gridcalBackend import GridCalBackend

# then some required things
from grid2op._create_test_suite import create_test_suite

def this_make_backend(self, detailed_infos_for_cascading_failures=False):
    return GridCalBackend(
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures
        )

add_name_cls = "test_GridCalBackend"

res = create_test_suite(make_backend_fun=this_make_backend,
                        add_name_cls=add_name_cls,
                        add_to_module=__name__,
                        extended_test=False,  # for now keep `extended_test=False` until all problems are solved
                        )

# and run it with `python -m unittest gridcal_backend_tests.py`
if __name__ == "__main__":
    unittest.main()
