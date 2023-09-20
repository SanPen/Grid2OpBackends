# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import copy
import numpy as np
import pandas as pd
from typing import Dict, Union

from grid2op.Backend.backend import Backend

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Backend.backend import Backend
from grid2op.Action import BaseAction
from grid2op.Exceptions import *

import newtonpa as npa

npa.findAndActivateLicense()


def decode_panda_structre(obj: Dict[str, str]) -> pd.DataFrame:
    """
    Decode the objects from pandapower json into an actual dataframe
    :param obj:
    :return:
    """
    data = json.loads(obj['_object'])

    return pd.DataFrame(data=data['data'], columns=data['columns'], index=data['index'])


def read_pandapower_file(filename: str) -> npa.HybridCircuit:
    """
    Parse pandapower json file into a GridCal MultiCircuit object
    :param filename: file path
    :return: MultiCircuit object
    """
    grid = npa.HybridCircuit()

    with open(filename) as f:
        data = json.load(f)
        data2 = data['_object']

    # buses
    bus_dict = dict()
    for i, row in decode_panda_structre(obj=data2['bus']).iterrows():
        name = 'sub_{}'.format(i)

        bus = npa.CalculationNode(uuid='',
                                  secondary_id='',
                                  name=name,
                                  active_default=int(row['in_service']),
                                  nominal_voltage=row['vn_kv'],
                                  vm_min=row['min_vm_pu'],
                                  vm_max=row['max_vm_pu'])
        bus_dict[i] = bus
        grid.addCalculationNode(bus)

    # loads
    for i, row in decode_panda_structre(obj=data2['load']).iterrows():
        bus = bus_dict[row['bus']]

        name = "load_{0}_{1}".format(row['bus'], i)

        load = npa.Load(uuid='',
                        secondary_id='',
                        name=name,
                        active_default=int(row['in_service']),
                        P=row['p_mw'] * row['scaling'],
                        Q=row['q_mvar'] * row['scaling'],
                        calc_node=bus)

        grid.addLoad(load)

    # shunt
    for i, row in decode_panda_structre(obj=data2['shunt']).iterrows():
        bus = bus_dict[row['bus']]

        cap = npa.Capacitor(uuid='',
                            secondary_id='',
                            name=str(row['name']),
                            active_default=int(row['in_service']),
                            G=row['p_mw'],
                            B=row['q_mvar'],
                            calc_node=bus)

        grid.addCapacitor(cap)

    # generators
    for i, row in decode_panda_structre(obj=data2['gen']).iterrows():

        bus = bus_dict[row['bus']]
        name = "gen_{0}_{1}".format(row['bus'], i)

        if row['slack']:
            bus.slack = True

        gen = npa.Generator(uuid='',
                            secondary_id='',
                            name=name,
                            active_default=int(row['in_service']),
                            P=row['p_mw'] * row['scaling'],
                            Vset=row['vm_pu'],
                            Pmin=row['min_p_mw'],
                            Pmax=row['max_p_mw'],
                            Qmin=row['min_q_mvar'],
                            Qmax=row['max_q_mvar'],
                            calc_node=bus)

        grid.addGenerator(gen)

    # lines
    for i, row in decode_panda_structre(obj=data2['line']).iterrows():
        bus_f = bus_dict[row['from_bus']]
        bus_t = bus_dict[row['to_bus']]

        name = 'CryLine {}'.format(i)

        line = npa.AcLine(uuid='',
                          secondary_id='',
                          name=name,
                          active_default=bool(row['in_service']),
                          calc_node_from=bus_f,
                          calc_node_to=bus_t)

        # "r_ohm", "x_ohm", "c_nf", "length", "Imax", "freq", "Sbase"
        line.fill_design_properties(r_ohm=row['r_ohm_per_km'],
                                    x_ohm=row['x_ohm_per_km'],
                                    c_nf=row['c_nf_per_km'],
                                    length=row['length_km'],
                                    Imax=row['max_i_ka'],
                                    freq=50.0,
                                    Sbase=grid.Sbase)
        grid.addAcLine(line)

    # transformer
    for i, row in decode_panda_structre(obj=data2['trafo']).iterrows():
        bus_f = bus_dict[row['lv_bus']]
        bus_t = bus_dict[row['hv_bus']]

        name = 'Line {}'.format(i)  # transformers are also calles Line apparently

        transformer = npa.Transformer2WFull(uuid='',
                                            secondary_id='',
                                            name=name,
                                            active_default=bool(row['in_service']),
                                            calc_node_from=bus_f,
                                            calc_node_to=bus_t,
                                            Vhigh=row['vn_hv_kv'],
                                            Vlow=row['vn_lv_kv'],
                                            rate=row['sn_mva'])

        transformer.fill_design_properties(Pcu=0.0,  # pandapower has no pcu apparently
                                           Pfe=row['pfe_kw'],
                                           I0=row['i0_percent'],
                                           Vsc=row['vk_percent'],
                                           Sbase=grid.Sbase)

        grid.addTransformers2wFul(transformer)

    # n = len(grid.get_generators())
    # n += len(grid.get_loads())
    # n += len(grid.get_shunts())
    # n += len(grid.get_batteries())
    # n += len(grid.get_branch_names_wo_hvdc()) * 2

    return grid


class NewtonBackend(Backend):
    """
    NewtonBackend
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

        # maximum number of iterations for the power flow
        self.max_iter = max_iter

        # use distributed slack?
        self.distributed_slack = dist_slack

        self._grid: Union[npa.HybridCircuit, None] = None
        self.numerical_circuit: Union[npa.NumericCircuit, None] = None
        self.results: Union[npa.PowerFlowResults, None] = None

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

        if self.results:
            theta = np.angle(self.results.voltage, deg=True)
            theta_f = self.numerical_circuit.branch_data.C_branch_bus_f * theta
            theta_t = self.numerical_circuit.branch_data.C_branch_bus_t * theta
            theta_load = theta * self.numerical_circuit.load_data.C
            theta_gen = theta * self.numerical_circuit.generator_data.C
            theta_bat = theta * self.numerical_circuit.battery_data.C
        else:
            theta_f = np.zeros(self.numerical_circuit.nbr)
            theta_t = np.zeros(self.numerical_circuit.nbr)
            theta_load = np.zeros(self.numerical_circuit.nload)
            theta_gen = np.zeros(self.numerical_circuit.ngen)
            theta_bat = np.zeros(self.numerical_circuit.nbatt)

        return theta_f, theta_t, theta_load, theta_gen, theta_bat

    def reset(self, path=None, grid_filename=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Reload the grid.
        """
        self.numerical_circuit = npa.compileAt(circuit=self._grid, t=0)

    def initiailize_or_update_internals(self):
        """
        Initialize / Update things after loading / copying
        """
        # and now initialize the attributes (see list bellow)
        self.n_line = self.numerical_circuit.branch_data.nelm  # number of lines in the grid should be read from self._grid
        self.n_gen = self.numerical_circuit.generator_data.nelm  # number of generators in the grid should be read from self._grid
        self.n_load = self.numerical_circuit.load_data.nelm  # number of generators in the grid should be read from self._grid
        self.n_sub = self.numerical_circuit.bus_data.nelm  # number of generators in the grid should be read from self._grid

        # other attributes should be read from self._grid (see table below for a full list of the attributes)
        self.load_to_subid = self.numerical_circuit.load_data.get_calc_node_indices()
        self.gen_to_subid = self.numerical_circuit.generator_data.get_calc_node_indices()
        self.line_or_to_subid = self.numerical_circuit.branch_data.F
        self.line_ex_to_subid = self.numerical_circuit.branch_data.T

        # naming
        self.name_load = self.numerical_circuit.load_data.names
        self.name_gen = self.numerical_circuit.generator_data.names
        self.name_line = self.numerical_circuit.branch_data.names
        self.name_sub = self.numerical_circuit.bus_data.names

        # and finish the initialization with a call to this function
        self._compute_pos_big_topo()

        # the initial thermal limit
        self.thermal_limit_a = self.numerical_circuit.branch_data.rates

    def load_grid(self, path=None, filename=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Load the _grid, and initialize all the member of the class. Note that in order to perform topological
        modification of the substation of the underlying powergrid, some buses are added to the test case loaded. They
        are set as "out of service" unless a topological action acts on these specific substations.

        """
        # parse the pandapower json file
        self._grid = read_pandapower_file(filename=path)

        # compile for easy numerical access
        self.numerical_circuit = npa.compileAt(circuit=self._grid, t=0)

        # initilize internals
        self.initiailize_or_update_internals()

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

        # TODO: sanpen: what is this?

        pass

    def apply_action(self, backendAction=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Specific implementation of the method to apply an action modifying a powergrid in the pandapower format.
        """
        if backendAction is None:
            return

        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backendAction()

        # buses
        for i, (bus1_status, bus2_status) in enumerate(active_bus):
            # TODO: what is this? why are there 2 buses?
            self.numerical_circuit.bus_data.setActiveAt(i, bus1_status)

        # generators
        for i, (changed, p) in enumerate(zip(prod_p.changed, prod_p.values)):
            if changed:
                self.numerical_circuit.generator_data.setPAt(i, p)

        # batteries
        for i, (changed, p) in enumerate(zip(storage.changed, storage.values)):
            if changed:
                self.numerical_circuit.battery_data.setPAt(i, p)

        # loads
        for i, (changed_p, p, changed_q, q) in enumerate(zip(load_p.changed, load_p.values,
                                                             load_q.changed, load_q.values)):
            if changed_p:
                self.numerical_circuit.load_data.setPAt(i, p)
            if changed_q:
                self.numerical_circuit.load_data.setQAt(i, q)

        # TODO: what about shunts?

        # TODO: what about topology?

    def runpf(self, is_dc=False):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Run a power flow on the underlying _grid. This implements an optimization of the powerflow
        computation: if the number of
        buses has not changed between two calls, the previous results are re used. This speeds up the computation
        in case of "do nothing" action applied.
        """

        """
        solver_type: newtonpa.SolverType = <SolverType.NR: 0>, 
        retry_with_other_methods: bool = True, 
        verbose: bool = False, 
        initialize_with_existing_solution: bool = False, 
        tolerance: float = 1e-06, 
        max_iter: int = 15, 
        control_q_mode: newtonpa.ReactivePowerControlMode = <ReactivePowerControlMode.NoControl: 0>, 
        tap_control_mode: newtonpa.TapsControlMode = <TapsControlMode.NoControl: 0>, 
        distributed_slack: bool = False, 
        ignore_single_node_islands: bool = False, 
        correction_parameter: float = 0.5, 
        mu0: float = 1.0) -> None
        """

        # power flow options
        pf_options = npa.PowerFlowOptions(max_iter=self.max_iter,
                                          distributed_slack=self.distributed_slack,
                                          solver_type=npa.SolverType.DC if is_dc else npa.SolverType.NR,
                                          retry_with_other_methods=True)

        # run the power flow
        self.results = npa.runSingleTimePowerFlow(numeric_circuit=self.numerical_circuit,
                                                  pf_options=pf_options,
                                                  V0=None)

        # return the status
        return self.results.converged, None

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
        # unattended copy
        tmp_ = self._grid.copy()
        self._grid = None
        res = copy.deepcopy(self)
        res._grid = tmp_
        self._grid = tmp_

        return res

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
        npa.FileHandler().save(circuit=self._grid, file_name=full_path)

    def get_line_status(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        As all the functions related to powerline,
        """
        return self.numerical_circuit.branch_data.active.astype(bool)

    def get_line_flow(self):
        return self.results.Sf.real[0, :]

    def _disconnect_line(self, id_):
        self.numerical_circuit.branch_data.active[id_] = 0

    def _reconnect_line(self, id_):
        self.numerical_circuit.branch_data.active[id_] = 1

    def get_topo_vect(self):
        # n = self.numerical_circuit.generator_data.nelm
        # n += self.numerical_circuit.load_data.nelm
        # n += self.numerical_circuit.shunt_data.nelm
        # n += self.numerical_circuit.branch_data.nelm * 2
        # n += self.numerical_circuit.battery_data.nelm
        # bus_idx, gen_idx, load_idx, shunt_idx, line_idx, storage_idx

        # declare a list for each bus
        data = {i: list() for i in range(self.numerical_circuit.bus_data.nelm)}

        # for each monopole structure ...
        for struct in [self.numerical_circuit.generator_data,
                       self.numerical_circuit.load_data,
                       # self.numerical_circuit.shunt_data  # shunts don't count
                       ]:
            for i, bus_idx in enumerate(struct.get_calc_node_indices()):
                data[bus_idx].append(i)

        # for each dipole structure ...
        for i, (f, t) in enumerate(zip(self.numerical_circuit.branch_data.F, self.numerical_circuit.branch_data.T)):
            data[f].append(i)
            data[t].append(t)

        # repeat for the battery data to have the same indexing ...
        for struct in [self.numerical_circuit.battery_data]:
            for i, bus_idx in enumerate(struct.get_calc_node_indices()):
                data[bus_idx].append(i)

        # arrange elements in order
        lst2 = list()
        for bus_idx in range(self.numerical_circuit.bus_data.nelm):
            lst2 += data[bus_idx]

        return np.array(lst2)

    def generators_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage[0, :] * self.numerical_circuit.bus_data.Vnom)
            P = self.numerical_circuit.generator_data.P
            Q = np.abs(self.results.Scalc[0, :].imag) * self.numerical_circuit.generator_data.C
            V = Vm * self.numerical_circuit.generator_data.C
        else:
            n = self.numerical_circuit.generator_data.nelm
            P = np.zeros(n)
            Q = np.zeros(n)
            V = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.generator_data.C

        return P, Q, V

    def loads_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage[0, :] * self.numerical_circuit.bus_data.Vnom)
            P = self.numerical_circuit.load_data.S.real
            Q = self.numerical_circuit.load_data.S.imag
            V = Vm * self.numerical_circuit.load_data.C

        else:
            n = self.numerical_circuit.load_data.nelm
            P = np.zeros(n)
            Q = np.zeros(n)
            V = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.load_data.C

        return P, Q, V

    def lines_or_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage[0, :] * self.numerical_circuit.bus_data.Vnom)
            P = self.results.Sf[0, :].real  # MW
            Q = self.results.Sf[0, :].imag  # MVAr
            V = self.numerical_circuit.branch_data.Cf * Vm  # kV
            A = self.numerical_circuit.branch_data.F

        else:
            n = self.numerical_circuit.branch_data.nelm
            P = np.zeros(n)  # MW
            Q = np.zeros(n)  # MVAr
            V = self.numerical_circuit.branch_data.Cf * self.numerical_circuit.bus_data.Vnom  # kV
            A = self.numerical_circuit.branch_data.F

        return P, Q, V, A

    def lines_ex_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage[0, :] * self.numerical_circuit.bus_data.Vnom)
            P = self.results.St[0, :].real  # MW
            Q = self.results.St[0, :].imag  # MVAr
            V = self.numerical_circuit.branch_data.Ct * Vm  # kV
            A = self.numerical_circuit.branch_data.T

        else:
            n = self.numerical_circuit.branch_data.nelm
            P = np.zeros(n)  # MW
            Q = np.zeros(n)  # MVAr
            V = self.numerical_circuit.branch_data.Ct * self.numerical_circuit.bus_data.Vnom  # kV
            A = self.numerical_circuit.branch_data.T

        return P, Q, V, A

    def shunt_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage[0, :] * self.numerical_circuit.bus_data.Vnom)
            P = self.results.Scalc.real[0, :] * self.numerical_circuit.shunt_data.C  # MW
            Q = self.results.Scalc.imag[0, :] * self.numerical_circuit.shunt_data.C  # MVAr
            V = Vm * self.numerical_circuit.shunt_data.C  # kV
            A = self.numerical_circuit.shunt_data.get_bus_indices()

        else:
            n = self.numerical_circuit.shunt_data.nelm
            P = np.zeros(n)  # MW
            Q = np.zeros(n)  # MVAr
            V = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.shunt_data.C  # kV
            A = self.numerical_circuit.shunt_data.get_bus_indices()

        return P, Q, V, A

    def storages_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage[0, :] * self.numerical_circuit.bus_data.Vnom)
            P = self.numerical_circuit.battery_data.p
            Q = np.abs(self.results.Scalc[0, :].imag) * self.numerical_circuit.battery_data.C
            V = Vm * self.numerical_circuit.battery_data.C
        else:
            n = self.numerical_circuit.battery_data.nelm
            P = np.zeros(n)
            Q = np.zeros(n)
            V = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.battery_data.C

        return P, Q, V

    def sub_from_bus_id(self, bus_id):
        # TODO: what to do here?
        # if bus_id >= self._number_true_line:
        #     return bus_id - self._number_true_line
        return bus_id
