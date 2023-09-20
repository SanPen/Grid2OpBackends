import unittest
import warnings

# first the backend class (for the example here)
from gridcalBackend import GridCalBackend

# then some required things
from grid2op.tests.helper_path_test import PATH_DATA_TEST_PP, PATH_DATA_TEST
from grid2op.tests.helper_path_test import HelperTests
PATH_DATA_TEST_INIT = PATH_DATA_TEST
PATH_DATA_TEST = PATH_DATA_TEST_PP

# then all the tests that can be automatically performed
from grid2op.tests.BaseBackendTest import BaseTestNames, BaseTestLoadingCase, BaseTestLoadingBackendFunc
from grid2op.tests.BaseBackendTest import BaseTestTopoAction, BaseTestEnvPerformsCorrectCascadingFailures
from grid2op.tests.BaseBackendTest import BaseTestChangeBusAffectRightBus, BaseTestShuntAction
from grid2op.tests.BaseBackendTest import BaseTestResetEqualsLoadGrid, BaseTestVoltageOWhenDisco, BaseTestChangeBusSlack
from grid2op.tests.BaseBackendTest import BaseIssuesTest, BaseStatusActions
from grid2op.tests.test_Environment import (TestLoadingBackendPandaPower as BaseTestLoadingBackendPandaPower,
                                            TestResetOk as BaseTestResetOk)
from grid2op.tests.test_Environment import (TestResetAfterCascadingFailure as TestResetAfterCascadingFailure,
                                            TestCascadingFailure as BaseTestCascadingFailure)
from grid2op.tests.BaseRedispTest import BaseTestRedispatch, BaseTestRedispatchChangeNothingEnvironment
from grid2op.tests.BaseRedispTest import BaseTestRedispTooLowHigh, BaseTestDispatchRampingIllegalETC
from grid2op.tests.BaseRedispTest import BaseTestLoadingAcceptAlmostZeroSumRedisp


class TestNames(HelperTests, BaseTestNames):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk

    def get_path(self):
        return PATH_DATA_TEST_INIT

