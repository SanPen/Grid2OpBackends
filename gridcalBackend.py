# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os  # load the python os default module
import sys  # laod the python sys default module
from typing import Dict
import json
import numpy as np
import pandas as pd
import scipy
from typing import Union

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Backend.backend import Backend
from grid2op.Action import BaseAction
from grid2op.Exceptions import *

import GridCalEngine.Core.Devices as dev
import GridCalEngine.Simulations as sim
from GridCalEngine.Core.Devices.multi_circuit import MultiCircuit
from GridCalEngine.Core.DataStructures.numerical_circuit import NumericalCircuit, compile_numerical_circuit_at


def decode_panda_structre(obj: Dict[str, str]) -> pd.DataFrame:
    """
    Decode the objects from pandapower json into an actual dataframe
    :param obj:
    :return:
    """
    data = json.loads(obj['_object'])

    return pd.DataFrame(data=data['data'], columns=data['columns'], index=data['index'])


class GridCalBackend(Backend):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        If you want to code a backend to use grid2op with another powerflow, you can get inspired
        from this class. Note However that implies knowing the behaviour
        of PandaPower.

    This module presents an example of an implementation of a `grid2op.Backend` when using the powerflow
    implementation "newton" available at `www.advancedgridinsights.com`_ for more details about
    this backend. This file is provided as an example of a proper :class:`grid2op.Backend.Backend` implementation.

    This backend currently does not work with 3 winding transformers and other exotic object.

    As explained in the `grid2op.Backend` module, every module must inherit the `grid2op.Backend` class.

    This class have more attributes that are used internally for faster information retrieval.

    Attributes
    ----------
    prod_pu_to_kv: :class:`numpy.array`, dtype:float
        The ratio that allow the conversion from pair-unit to kv for the generators

    load_pu_to_kv: :class:`numpy.array`, dtype:float
        The ratio that allow the conversion from pair-unit to kv for the loads

    lines_or_pu_to_kv: :class:`numpy.array`, dtype:float
        The ratio that allow the conversion from pair-unit to kv for the origin end of the powerlines

    lines_ex_pu_to_kv: :class:`numpy.array`, dtype:float
        The ratio that allow the conversion from pair-unit to kv for the extremity end of the powerlines

    p_or: :class:`numpy.array`, dtype:float
        The active power flowing at the origin end of each powerline

    q_or: :class:`numpy.array`, dtype:float
        The reactive power flowing at the origin end of each powerline

    v_or: :class:`numpy.array`, dtype:float
        The voltage magnitude at the origin bus of the powerline

    a_or: :class:`numpy.array`, dtype:float
        The current flowing at the origin end of each powerline

    p_ex: :class:`numpy.array`, dtype:float
        The active power flowing at the extremity end of each powerline

    q_ex: :class:`numpy.array`, dtype:float
        The reactive power flowing at the extremity end of each powerline

    a_ex: :class:`numpy.array`, dtype:float
        The current flowing at the extremity end of each powerline

    v_ex: :class:`numpy.array`, dtype:float
        The voltage magnitude at the extremity bus of the powerline

    Examples
    ---------
    The only recommended way to use this class is by passing an instance of a Backend into the "make"
    function of grid2op. Do not attempt to use a backend outside of this specific usage.

    .. code-block:: python

            import grid2op
            from grid2op.Backend import PandaPowerBackend
            backend = PandaPowerBackend()

            env = grid2op.make(backend=backend)
            # and use "env" as any open ai gym environment.

    """

    def __init__(self,
                 detailed_infos_for_cascading_failures=False,
                 dist_slack=False,
                 max_iter=10):
        """

        """

        Backend.__init__(self,
                         detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
                         can_be_copied=True,
                         lightsim2grid=False,
                         dist_slack=dist_slack,
                         max_iter=max_iter)

        self.prod_pu_to_kv = None
        self.load_pu_to_kv = None
        self.lines_or_pu_to_kv = None
        self.lines_ex_pu_to_kv = None
        self.storage_pu_to_kv = None

        self.p_or = None
        self.q_or = None
        self.v_or = None
        self.a_or = None
        self.p_ex = None
        self.q_ex = None
        self.v_ex = None
        self.a_ex = None

        self.load_p = None
        self.load_q = None
        self.load_v = None

        self.storage_p = None
        self.storage_q = None
        self.storage_v = None

        self.prod_p = None
        self.prod_q = None
        self.prod_v = None
        self.line_status = None

        self._pf_init = "flat"
        self._pf_init = "results"
        self._nb_bus_before = None  # number of active bus at the preceeding step

        self.thermal_limit_a = None

        self._iref_slack = None
        self._id_bus_added = None
        self._fact_mult_gen = -1
        self._what_object_where = None
        self._number_true_line = -1
        self._corresp_name_fun = {}
        self._get_vector_inj = {}
        self.dim_topo = -1
        self._vars_action = BaseAction.attr_list_vect
        self._vars_action_set = BaseAction.attr_list_vect
        self.cst_1 = dt_float(1.0)
        self._topo_vect = None
        self.slack_id = None

        # function to rstore some information
        self.__nb_bus_before = None  # number of substation in the powergrid
        self.__nb_powerline = (
            None  # number of powerline (real powerline, not transformer)
        )
        self._init_bus_load = None
        self._init_bus_gen = None
        self._init_bus_lor = None
        self._init_bus_lex = None
        self._get_vector_inj = None
        self._big_topo_to_obj = None
        self._big_topo_to_backend = None
        self.__pp_backend_initial_grid = None  # initial state to facilitate the "reset"

        # Mapping some fun to apply bus updates
        self._type_to_bus_set = [
            # self._apply_load_bus,
            # self._apply_gen_bus,
            # self._apply_lor_bus,
            # self._apply_trafo_hv,
            # self._apply_lex_bus,
            # self._apply_trafo_lv,
        ]

        self.tol = None  # this is NOT the pandapower tolerance !!!! this is used to check if a storage unit
        # produce / absorbs anything

        # TODO storage doc (in grid2op rst) of the backend
        self.can_output_theta = True  # I support the voltage angle
        self.theta_or = None
        self.theta_ex = None
        self.load_theta = None
        self.gen_theta = None
        self.storage_theta = None

        self._dist_slack = dist_slack
        self._max_iter = max_iter

        self.pf_options = sim.PowerFlowOptions(max_iter=max_iter)
        self._grid: Union[MultiCircuit, None] = None
        self.numerical_circuit: Union[NumericalCircuit, None] = None
        self.results: Union[sim.PowerFlowResults, None] = None

    def get_theta(self):
        """
        Returns the voltage angle in degrees from several devices

        Returns
        -------
        theta_or: ``numpy.ndarray``
            For each orgin side of powerline, gives the voltage angle (in degree)
        theta_ex: ``numpy.ndarray``
            For each extremity side of powerline, gives the voltage angle (in degree)
        load_theta: ``numpy.ndarray``
            Gives the voltage angle (in degree) to the bus at which each load is connected
        gen_theta: ``numpy.ndarray``
            Gives the voltage angle (in degree) to the bus at which each generator is connected
        storage_theta: ``numpy.ndarray``
            Gives the voltage angle (in degree) to the bus at which each storage unit is connected
        """

        return (
            self.cst_1 * self.theta_or,
            self.cst_1 * self.theta_ex,
            self.cst_1 * self.load_theta,
            self.cst_1 * self.gen_theta,
            self.cst_1 * self.storage_theta,
        )

    def reset(self, path=None, grid_filename=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Reload the grid.
        """
        self.numerical_circuit = compile_numerical_circuit_at(circuit=self._grid, t_idx=None)

    def load_grid(self, path=None, filename=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Load the _grid, and initialize all the member of the class. Note that in order to perform topological
        modification of the substation of the underlying powergrid, some buses are added to the test case loaded. They
        are set as "out of service" unless a topological action acts on these specific substations.

        """
        self._grid = MultiCircuit()

        with open(path) as f:
            data = json.load(f)
            data2 = data['_object']

        # buses
        bus_dict = dict()
        for i, row in decode_panda_structre(obj=data2['bus']).iterrows():
            bus = dev.Bus(idtag='',
                          code='',
                          name=str(row['name']),
                          active=True,
                          vnom=row['vn_kv'],
                          vmin=row['min_vm_pu'],
                          vmax=row['max_vm_pu'])
            bus_dict[i] = bus
            self._grid.add_bus(bus)

        # loads
        for i, row in decode_panda_structre(obj=data2['load']).iterrows():
            bus = bus_dict[row['bus']]
            self._grid.add_load(bus, dev.Load(idtag='',
                                              code='',
                                              name=str(row['name']),
                                              active=True,
                                              P=row['p_mw'] * row['scaling'],
                                              Q=row['q_mvar'] * row['scaling']))

        # shunt
        for i, row in decode_panda_structre(obj=data2['shunt']).iterrows():
            bus = bus_dict[row['bus']]
            self._grid.add_shunt(bus, dev.Shunt(idtag='',
                                                code='',
                                                name=str(row['name']),
                                                active=True,
                                                G=row['p_mw'],
                                                B=row['q_mvar']))

        # generators
        for i, row in decode_panda_structre(obj=data2['gen']).iterrows():

            bus = bus_dict[row['bus']]

            if row['slack']:
                bus.is_slack = True

            self._grid.add_generator(bus, dev.Generator(idtag='',
                                                        code='',
                                                        name=str(row['name']),
                                                        active=True,
                                                        active_power=row['p_mw'] * row['scaling'],
                                                        voltage_module=row['vm_pu'],
                                                        p_min=row['min_p_mw'],
                                                        p_max=row['max_p_mw'],
                                                        Qmin=row['min_q_mvar'],
                                                        Qmax=row['max_q_mvar'], ))

        # lines
        for i, row in decode_panda_structre(obj=data2['line']).iterrows():
            bus_f = bus_dict[row['from_bus']]
            bus_t = bus_dict[row['to_bus']]
            line = dev.Line(idtag='',
                            code='',
                            name=str(row['name']),
                            active=True,
                            bus_from=bus_f,
                            bus_to=bus_t)

            line.fill_design_properties(r_ohm=0,
                                        x_ohm=0,
                                        c_nf=0,
                                        Imax=0,
                                        freq=50,
                                        length=0,
                                        Sbase=self._grid.Sbase)
            self._grid.add_line(line)

        # transformer
        for i, row in decode_panda_structre(obj=data2['trafo']).iterrows():
            bus_f = bus_dict[row['from_bus']]
            bus_t = bus_dict[row['to_bus']]

            transformer = dev.Transformer2W(idtag='',
                                            code='',
                                            name=str(row['name']),
                                            active=True,
                                            bus_from=bus_f,
                                            bus_to=bus_t)

            transformer.fill_design_properties(Pcu=0,
                                               Pfe=0,
                                               I0=0,
                                               Vsc=0,
                                               Sbase=self._grid.Sbase)

            self._grid.add_transformer2w(transformer)

        self.numerical_circuit = compile_numerical_circuit_at(circuit=self._grid, t_idx=None)

    def storage_deact_for_backward_comaptibility(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This function is called under a very specific condition: an old environment has been loaded that
        do not take into account the storage units, even though they were possibly some modeled by the backend.

        This function is supposed to "remove" from the backend any reference to the storage units.

        Overloading this function is not necessary (when developing a new backend). If it is not overloaded however,
        some "backward compatibility" (for grid2op <= 1.4.0) might not be working properly depending on
        your backend.
        """

        # TODO: sanpen: WTF is this?

        pass

    def apply_action(self, backendAction=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Specific implementation of the method to apply an action modifying a powergrid in the pandapower format.
        """
        if backendAction is None:
            return

        cls = type(self)

        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backendAction()

        # handle bus status
        bus_is = self._grid.bus["in_service"]
        # TODO: what is this? why are there 2 buses?
        for i, (bus1_status, bus2_status) in enumerate(active_bus):
            bus_is[i] = bus1_status  # no iloc for bus, don't ask me why please :-/
            bus_is[i + self.__nb_bus_before] = bus2_status
            # self.numerical_circuit.bus_data.setActiveAt()

        # tmp_prod_p = self._get_vector_inj["prod_p"](self._grid)
        if prod_p.changed.any():
            # tmp_prod_p.iloc[prod_p.changed] = prod_p.values[prod_p.changed]
            self.numerical_circuit.generator_data.P = prod_p.values

        tmp_prod_v = self._get_vector_inj["prod_v"](self._grid)
        if prod_v.changed.any():
            # tmp_prod_v.iloc[prod_v.changed] = (
            #         prod_v.values[prod_v.changed] / self.prod_pu_to_kv[prod_v.changed]
            # )
            self.numerical_circuit.generator_data.vset = prod_v.values

        # todo: add buses on the fly? highly sus
        if self._id_bus_added is not None and prod_v.changed[self._id_bus_added]:
            # handling of the slack bus, where "2" generators are present.
            self._grid["ext_grid"]["vm_pu"] = 1.0 * tmp_prod_v[self._id_bus_added]

        if load_p.changed.any():
            self.numerical_circuit.load_data.P = load_p.values

        if load_q.changed.any():
            self.numerical_circuit.load_data.Q = load_q.values

        if self.n_storage:

            # active setpoint
            if storage.changed.any():
                self.numerical_circuit.battery_data.P = storage.values

            # topology of the storage
            stor_bus = backendAction.get_storages_bus()
            new_bus_id = stor_bus.values[stor_bus.changed]  # id of the busbar 1 or 2 if
            activated = new_bus_id > 0  # mask of storage that have been activated
            new_bus_num = self.storage_to_subid[stor_bus.changed] + (new_bus_id - 1) * self.n_sub  # bus number
            new_bus_num[~activated] = self.storage_to_subid[stor_bus.changed][~activated]
            self._grid.storage["in_service"].values[stor_bus.changed] = activated
            self._grid.storage["bus"].values[stor_bus.changed] = new_bus_num
            self._topo_vect[self.storage_pos_topo_vect[stor_bus.changed]] = new_bus_num
            self._topo_vect[self.storage_pos_topo_vect[stor_bus.changed][~activated]] = -1

        if type(backendAction).shunts_data_available:
            shunt_p, shunt_q, shunt_bus = shunts__

            if (shunt_p.changed).any():
                self._grid.shunt["p_mw"].iloc[shunt_p.changed] = shunt_p.values[
                    shunt_p.changed
                ]
            if (shunt_q.changed).any():
                self._grid.shunt["q_mvar"].iloc[shunt_q.changed] = shunt_q.values[
                    shunt_q.changed
                ]
            if (shunt_bus.changed).any():
                sh_service = shunt_bus.values[shunt_bus.changed] != -1
                self._grid.shunt["in_service"].iloc[shunt_bus.changed] = sh_service
                chg_and_in_service = sh_service & shunt_bus.changed
                self._grid.shunt["bus"].loc[chg_and_in_service] = cls.local_bus_to_global(
                    shunt_bus.values[chg_and_in_service],
                    cls.shunt_to_subid[chg_and_in_service])

        # i made at least a real change, so i implement it in the backend
        for id_el, new_bus in topo__:
            id_el_backend, id_topo, type_obj = self._big_topo_to_backend[id_el]

            if type_obj is not None:
                # storage unit are handled elsewhere
                self._type_to_bus_set[type_obj](new_bus, id_el_backend, id_topo)

    def runpf(self, is_dc=False):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Run a power flow on the underlying _grid. This implements an optimization of the powerflow
        computation: if the number of
        buses has not changed between two calls, the previous results are re used. This speeds up the computation
        in case of "do nothing" action applied.
        """
        newtonpa.runSingleTimePowerFlow(numeric_circuit=self.numerical_circuit,
                                        pf_options=self.pf_options,
                                        V0=None)

    def assert_grid_correct(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is done as it should be by the Environment
        """
        super().assert_grid_correct()

    def copy(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Performs a deep copy of the power :attr:`_grid`.
        """
        return self._grid.copy()

    def close(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Called when the :class:`grid2op;Environment` has terminated, this function only reset the grid to a state
        where it has not been loaded.
        """
        self._grid = None
        self.numerical_circuit = None

    def save_file(self, full_path):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        You might want to use it for debugging purpose only, and only if you develop yourself a backend.

        Save the file to json.
        :param full_path:
        :return:
        """
        newtonpa.FileHandler().save(circuit=self._grid, file_name=full_path)

    def get_line_status(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        As all the functions related to powerline,
        """
        return self.numerical_circuit.branch_data.active

    def get_line_flow(self):
        return self.results.Sf

    def _disconnect_line(self, id_):
        self.numerical_circuit.branch_data.active[id_] = 0

    def _reconnect_line(self, id_):
        self.numerical_circuit.branch_data.active[id_] = 1

    def get_topo_vect(self):
        return self._topo_vect

    def generators_info(self):
        return (
            self.cst_1 * self.prod_p,
            self.cst_1 * self.prod_q,
            self.cst_1 * self.prod_v,
        )

    def loads_info(self):
        return (
            self.cst_1 * self.load_p,
            self.cst_1 * self.load_q,
            self.cst_1 * self.load_v,
        )

    def lines_or_info(self):
        return (
            self.cst_1 * self.p_or,
            self.cst_1 * self.q_or,
            self.cst_1 * self.v_or,
            self.cst_1 * self.a_or,
        )

    def lines_ex_info(self):
        return (
            self.cst_1 * self.p_ex,
            self.cst_1 * self.q_ex,
            self.cst_1 * self.v_ex,
            self.cst_1 * self.a_ex,
        )

    def shunt_info(self):
        shunt_p = self.cst_1 * self._grid.res_shunt["p_mw"].values.astype(dt_float)
        shunt_q = self.cst_1 * self._grid.res_shunt["q_mvar"].values.astype(dt_float)
        shunt_v = (
            self._grid.res_bus["vm_pu"]
            .loc[self._grid.shunt["bus"].values]
            .values.astype(dt_float)
        )
        shunt_v *= (
            self._grid.bus["vn_kv"]
            .loc[self._grid.shunt["bus"].values]
            .values.astype(dt_float)
        )
        shunt_bus = type(self).global_bus_to_local(self._grid.shunt["bus"].values, self.shunt_to_subid)
        shunt_v[~self._grid.shunt["in_service"].values] = 0
        shunt_bus[~self._grid.shunt["in_service"].values] = -1
        # handle shunt alone on a bus (in this case it should probably diverge...)
        alone = ~np.isfinite(shunt_v)
        shunt_v[alone] = 0
        shunt_bus[alone] = -1
        return shunt_p, shunt_q, shunt_v, shunt_bus

    def storages_info(self):
        return (
            self.cst_1 * self.storage_p,
            self.cst_1 * self.storage_q,
            self.cst_1 * self.storage_v,
        )

    def sub_from_bus_id(self, bus_id):
        if bus_id >= self._number_true_line:
            return bus_id - self._number_true_line
        return bus_id
