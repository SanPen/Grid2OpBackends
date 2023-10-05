import unittest
import warnings

# first the backend class (for the example here)
from gridcalBackend import GridCalBackend

# then some required things
from grid2op.tests.helper_path_test import PATH_DATA_TEST_PP, PATH_DATA_TEST
from grid2op.tests.helper_path_test import HelperTests
PATH_DATA_TEST_INIT = PATH_DATA_TEST
PATH_DATA_TEST = PATH_DATA_TEST_PP

from grid2op.Space import GridObjects  # lazy import
__has_storage = hasattr(GridObjects, "n_storage")

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
if __has_storage:
    from grid2op.tests.BaseBackendTest import BaseTestStorageAction


class TestNamesGCBk(HelperTests, BaseTestNames):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk

    def get_path(self):
        return PATH_DATA_TEST_INIT


class TestLoadingCaseGCBk(HelperTests, BaseTestLoadingCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestLoadingBackendFuncGCBk(HelperTests, BaseTestLoadingBackendFunc):
    def setUp(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.setUp(self)
        self.tests_skipped = set()

        # lightsim does not support DC powerflow at the moment
        # self.tests_skipped.add("test_pf_ac_dc")
        # self.tests_skipped.add("test_apply_action_active_value")
        # self.tests_skipped.add("test_runpf_dc")
        # Now (version >= 0.5.5) it does

    def tearDown(self):
        # TODO find something more elegant
        BaseTestLoadingBackendFunc.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestTopoActionGCBk(HelperTests, BaseTestTopoAction):
    def setUp(self):
        BaseTestTopoAction.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestTopoAction.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk

    def get_path(self):
        return PATH_DATA_TEST

    def get_casefile(self):
        return "test_case14.json"


class TestEnvPerformsCorrectCascadingFailuresGCBk(HelperTests, BaseTestEnvPerformsCorrectCascadingFailures):
    def setUp(self):
        BaseTestEnvPerformsCorrectCascadingFailures.setUp(self)

    def tearDown(self):
        # TODO find something more elegant
        BaseTestEnvPerformsCorrectCascadingFailures.tearDown(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk

    def get_casefile(self):
        return "test_case14.json"

    def get_path(self):
        return PATH_DATA_TEST


class TestChangeBusAffectRightBusGCBk(HelperTests, BaseTestChangeBusAffectRightBus):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk


class TestShuntActionGCBk(HelperTests, BaseTestShuntAction):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk


class TestResetEqualsLoadGridGCBk(HelperTests, BaseTestResetEqualsLoadGrid):
    def setUp(self):
        BaseTestResetEqualsLoadGrid.setUp(self)

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk


class TestVoltageOWhenDiscoGCBk(HelperTests, BaseTestVoltageOWhenDisco):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk


class TestChangeBusSlackGCBk(HelperTests, BaseTestChangeBusSlack):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk


class TestIssuesTestGCBk(HelperTests, BaseIssuesTest):
    tests_skipped = []
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk


class TestStatusActionGCBk(HelperTests, BaseStatusActions):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return bk


if __has_storage:
    class TestStorageActionGCBk(HelperTests, BaseTestStorageAction):
        def setUp(self):
            self.tests_skipped = ["test_storage_action_topo"]  # TODO this test is super weird ! It's like we impose
            # TODO a behaviour from pandapower (weird one) to all backends...

        def make_backend(self, detailed_infos_for_cascading_failures=False):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                bk = GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
            return bk


class TestLoadingBackendGCBk(BaseTestLoadingBackendPandaPower):
    def get_backend(self, detailed_infos_for_cascading_failures=True):
        return GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestResetOkGCBk(BaseTestResetOk):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestResetAfterCascadingFailureGCBk(TestResetAfterCascadingFailure):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestCascadingFailureGCBk(BaseTestCascadingFailure):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return GridCalBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


if __name__ == "__main__":
    unittest.main()
