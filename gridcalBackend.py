# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import json
import copy
import warnings

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union

from grid2op.Backend.backend import Backend
import pandapower as pp

import GridCalEngine.Core.Devices as dev
import GridCalEngine.Simulations as sim
from GridCalEngine.Core.Devices.multi_circuit import MultiCircuit
from GridCalEngine.Core.DataStructures.numerical_circuit import NumericalCircuit, compile_numerical_circuit_at
from GridCalEngine.Simulations.PowerFlow.power_flow_worker import multi_island_pf_nc
from GridCalEngine.basic_structures import SolverType
from GridCalEngine.IO.file_handler import FileSave


class BusActivationException(Exception):
    pass

def decode_panda_structre(obj: Dict[str, str]) -> pd.DataFrame:
    """
    Decode the objects from pandapower json into an actual dataframe
    :param obj:
    :return:
    """
    data = json.loads(obj['_object'])

    return pd.DataFrame(data=data['data'], columns=data['columns'], index=data['index'])


def _line_name(row, id_obj):
    return f"{row['from_bus']}_{row['to_bus']}_{id_obj}"


def _trafo_name(row, id_obj):
    return f"{row['hv_bus']}_{row['lv_bus']}_{id_obj}"


def _gen_name(row, id_obj):
    return f"gen_{row['bus']}_{id_obj}"


def _load_name(row, id_obj):
    return f"load_{row['bus']}_{id_obj}"


def _storage_name(row, id_obj):
    return f"storage_{row['bus']}_{id_obj}"


def _sub_name(row, id_obj):
    return f"sub_{id_obj}"


def _shunt_name(row, id_obj):
    return f"shunt_{row['bus']}_{id_obj}"


def aux_get_names(grid, grid_attrs):
    res = []
    obj_id = 0
    for (attr, fun_to_name) in grid_attrs:
        df = getattr(grid, attr)
        if (
                "name" in df.columns
                and not df["name"].isnull().values.any()
        ):
            res += [name for name in df["name"]]
        else:
            res += [
                fun_to_name(row, id_obj=obj_id + i)
                for i, (_, row) in enumerate(df.iterrows())
            ]
            obj_id += df.shape[0]
    res = np.array(res)
    return res


def read_pandapower_file(filename: str) -> MultiCircuit:
    """
    Parse pandapower json file into a GridCal MultiCircuit object
    :param filename: file path
    :return: MultiCircuit object
    """
    grid = MultiCircuit()

    # load the grid using pandapower (just for loading utilities, as they store things in some weird format)
    with warnings.catch_warnings():
        # remove deprecationg warnings for old version of pandapower
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        pp_grid = pp.from_json(filename)

    # buses
    # sub_names = aux_get_names(pp_grid, [("bus", _sub_name)])
    sub_names = ["sub_{}".format(i) for i, row in pp_grid.bus.iterrows()]
    bus_dict_by_index = dict()
    grid.substations = []
    for i in range(pp_grid.bus.values.shape[0]):

        name = sub_names[i]
        idx = pp_grid.bus.index[i]
        vmin = pp_grid.bus.min_vm_pu[i] if 'min_vm_pu' in pp_grid.bus else 0.8 * pp_grid.bus.vn_kv[i]
        vmax = pp_grid.bus.max_vm_pu[i] if 'max_vm_pu' in pp_grid.bus else 1.2 * pp_grid.bus.vn_kv[i]

        # create a new substation
        sub = dev.Substation(name='sub_{}'.format(idx))

        # create the bus
        bus = dev.Bus(idtag='',
                      code='1',
                      name='sub_{}'.format(idx),
                      active=bool(pp_grid.bus.in_service[i]),
                      vnom=pp_grid.bus.vn_kv[i],
                      vmin=vmin,
                      vmax=vmax,
                      substation=sub
                      )

        bus_dict_by_index[idx] = bus
        grid.add_bus(bus)
        grid.add_substation(sub)

    # loads
    load_names = aux_get_names(pp_grid, [("load", _load_name)])
    for i, row in pp_grid.load.iterrows():
        bus = bus_dict_by_index[row['bus']]

        name = load_names[i]

        grid.add_load(bus, dev.Load(idtag='',
                                    code='',
                                    name=name,
                                    active=bool(row['in_service']),
                                    P=row['p_mw'] * row['scaling'],
                                    Q=row['q_mvar'] * row['scaling']))

    # shunt
    shunt_names = aux_get_names(pp_grid, [("shunt", _shunt_name)])
    for i, row in pp_grid.shunt.iterrows():
        bus = bus_dict_by_index[row['bus']]
        grid.add_shunt(bus, dev.Shunt(idtag='',
                                      code='',
                                      name=shunt_names[i],
                                      active=bool(row['in_service']),
                                      G=row['p_mw'],
                                      B=row['q_mvar']))

    # generators
    gen_names = aux_get_names(pp_grid, [("gen", _gen_name)])
    for i, row in pp_grid.gen.iterrows():

        bus = bus_dict_by_index[row['bus']]
        name = gen_names[i]

        if row['slack']:
            bus.is_slack = True

        Pmin = row['min_p_mw'] if 'min_p_mw' in row else None
        Pmax = row['max_p_mw'] if 'max_p_mw' in row else None
        Qmin = row['min_q_mvar'] if 'min_q_mvar' in row else None
        Qmax = row['max_q_mvar'] if 'max_q_mvar' in row else None
        if Pmin is None:
            Pmin = -999999.
        if Pmax is None:
            Pmax = 999999.
        if Qmin is None:
            Qmin = -999999.
        if Qmax is None:
            Qmax = 999999.
        grid.add_generator(bus, dev.Generator(idtag='',
                                              code='',
                                              name=name,
                                              active=bool(row['in_service']),
                                              P=row['p_mw'] * row['scaling'],
                                              vset=row['vm_pu'],
                                              Pmin=Pmin,
                                              Pmax=Pmax,
                                              Qmin=Qmin,
                                              Qmax=Qmax, ))

    # ext_grids
    ext_grid_names = aux_get_names(pp_grid, [("ext_grid", _gen_name)])
    for i, row in pp_grid.ext_grid.iterrows():
        bus = bus_dict_by_index[row['bus']]
        name = ext_grid_names[i]
        bus.is_slack = True
        Pmin = -999999.
        Pmax = 999999.
        Qmin = -999999.
        Qmax = 999999.
        grid.add_generator(bus, dev.Generator(idtag='',
                                              code='',
                                              name=name,
                                              active=bool(row['in_service']),
                                              P=0.0,
                                              vset=row['vm_pu'],
                                              Pmin=Pmin,
                                              Pmax=Pmax,
                                              Qmin=Qmin,
                                              Qmax=Qmax, ))

    # lines
    line_names = aux_get_names(pp_grid, [("line", _line_name), ("trafo", _trafo_name)])
    for i, row in pp_grid.line.iterrows():
        bus_f = bus_dict_by_index[row['from_bus']]
        bus_t = bus_dict_by_index[row['to_bus']]

        # name = 'CryLine {}'.format(i)
        name = line_names[i]

        line = dev.Line(idtag='',
                        code='',
                        name=name,
                        active=bool(row['in_service']),
                        bus_from=bus_f,
                        bus_to=bus_t)

        line.fill_design_properties(r_ohm=row['r_ohm_per_km'],
                                    x_ohm=row['x_ohm_per_km'],
                                    c_nf=row['c_nf_per_km'],
                                    Imax=row['max_i_ka'],
                                    freq=pp_grid.f_hz,
                                    length=row['length_km'],
                                    Sbase=grid.Sbase)
        grid.add_line(line)

    # transformer
    for i, row in pp_grid.trafo.iterrows():
        bus_f = bus_dict_by_index[row['lv_bus']]
        bus_t = bus_dict_by_index[row['hv_bus']]

        # name = 'Line {}'.format(i)  # transformers are also calles Line apparently
        name = line_names[i + pp_grid.line.shape[0]]

        transformer = dev.Transformer2W(idtag='',
                                        code='',
                                        name=name,
                                        active=bool(row['in_service']),
                                        bus_from=bus_f,
                                        bus_to=bus_t,
                                        HV=row['vn_hv_kv'],
                                        LV=row['vn_lv_kv'],
                                        rate=row['sn_mva'])

        transformer.fill_design_properties(Pcu=0.0,  # pandapower has no pcu apparently
                                           Pfe=row['pfe_kw'],
                                           I0=row['i0_percent'],
                                           Vsc=row['vk_percent'],
                                           Sbase=grid.Sbase)

        grid.add_transformer2w(transformer)

    # n = len(grid.get_generators())
    # n += len(grid.get_loads())
    # n += len(grid.get_shunts())
    # n += len(grid.get_batteries())
    # n += len(grid.get_branch_names_wo_hvdc()) * 2

    return grid


class GridCalBackend(Backend):
    """
    Grid2Op backend using GridCalEngine
    """

    # required to be defined
    shunts_data_available = True

    def __init__(self,
                 detailed_infos_for_cascading_failures=False,
                 dist_slack=False,
                 max_iter=10):
        """

        :param detailed_infos_for_cascading_failures:
        :param dist_slack:
        :param max_iter:
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

        # GridCal main circuit object
        self._grid: Union[MultiCircuit, None] = None

        # GridCal numerical circuit object for easy numerical access / modification
        self.numerical_circuit: Union[NumericalCircuit, None] = None

        # Power flow results
        self.results: Union[sim.PowerFlowResults, None] = None

        # auxiliary stuff
        self.generator_bus_indices: Union[np.ndarray, None] = None
        self.generator_bus_nominal_voltages: Union[np.ndarray, None] = None

        self.bus_idx_in_sub = dict()
        
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
            theta_load = theta * self.numerical_circuit.load_data.C_bus_elm
            theta_gen = theta * self.numerical_circuit.generator_data.C_bus_elm
            theta_bat = theta * self.numerical_circuit.battery_data.C_bus_elm
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
        self.numerical_circuit = compile_numerical_circuit_at(circuit=self._grid, t_idx=None)

    def initiailize_or_update_internals(self):
        """
        Initialize / Update things after loading / copying
        """
        # and now initialize the attributes (see list bellow)
        self.n_line = self.numerical_circuit.nbr  # number of lines in the grid should be read from self._grid
        self.n_gen = self.numerical_circuit.ngen  # number of generators in the grid should be read from self._grid
        self.n_load = self.numerical_circuit.nload  # number of generators in the grid should be read from self._grid
        self.n_sub = self.numerical_circuit.nbus  # number of substations in the grid should be read from self._grid
        self.n_shunt = self.numerical_circuit.nshunt  # number of shunts in the grid 

        # other attributes should be read from self._grid (see table below for a full list of the attributes)
        self.load_to_subid = self.numerical_circuit.load_data.get_bus_indices()
        self.gen_to_subid = self.numerical_circuit.generator_data.get_bus_indices()
        self.line_or_to_subid = self.numerical_circuit.branch_data.F
        self.line_ex_to_subid = self.numerical_circuit.branch_data.T
        self.shunt_to_subid = self.numerical_circuit.shunt_data.get_bus_indices()

        # naming
        self.name_load = self.numerical_circuit.load_data.names
        self.name_gen = self.numerical_circuit.generator_data.names
        self.name_line = self.numerical_circuit.branch_data.names
        self.name_sub = self.numerical_circuit.bus_data.names
        self.name_shunt = self.numerical_circuit.shunt_data.names

        # and finish the initialization with a call to this function
        self._compute_pos_big_topo()

        # Trick for accessing object method when the class attribute is not available
        self.shunts_data_available = True

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

        if path is None and filename is None:
            raise RuntimeError(
                "You must provide at least one of path or file to load a powergrid."
            )
        if path is None:
            full_path = filename
        elif filename is None:
            full_path = path
        else:
            full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            raise RuntimeError('There is no powergrid at "{}"'.format(full_path))

        # parse the pandapower json file
        self._grid = read_pandapower_file(filename=full_path)

        # compile for easy numerical access
        self.numerical_circuit = compile_numerical_circuit_at(circuit=self._grid, t_idx=None)

        self.generator_bus_nominal_voltages = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.generator_data.C_bus_elm
        self.generator_bus_indices = self.numerical_circuit.generator_data.get_bus_indices()

        # initilize internals
        self.initiailize_or_update_internals()

        # Map buses to code
        self.bus_idx_in_sub = dict()
        for i, bus in enumerate(self._grid.buses):
            self.bus_idx_in_sub[i] = 1 if bus.code == "1" else 2

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
        # Ben: not to be implemented, it's a real internal function.

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
        subs_vect = self.numerical_circuit.bus_data.substations
        for i, (bus1_status, bus2_status) in enumerate(active_bus):
            # TODO: what is this? why are there 2 buses? Maybe the iteration is over substations
            sub_vect= np.where(subs_vect == i)[0]
            if len(sub_vect) == 1:
                bus1_id = sub_vect[0]
                self.numerical_circuit.bus_data.active[bus1_id] = int(bus1_status)
            elif len(sub_vect) == 2:
                bus1_id, bus2_id = sub_vect
                self.numerical_circuit.bus_data.active[bus1_id] = int(bus1_status)
                self.numerical_circuit.bus_data.active[bus2_id] = int(bus2_status)
            else:
                raise BusActivationException("There was a problem")

        # generators
        gen_bus = backendAction.get_gens_bus()
        for i, (changed_p, p,
                changed_v, vset,
                bus_changed, bus_value) in enumerate(zip(prod_p.changed, prod_p.values,
                                                         prod_v.changed, prod_v.values,
                                                         gen_bus.changed, gen_bus.values)):
            if changed_p:
                self.numerical_circuit.generator_data.p[i] = p  # in MW, I hope

            if changed_v:
                # vset comes in kV, we need the p.u. set point
                vnom = self.generator_bus_nominal_voltages[i]
                self.numerical_circuit.generator_data.v[i] = vset / vnom

            if bus_changed:
                self.numerical_circuit.generator_data.active[i] = int(bus_value != -1)

        # Alternative:
        # generators' power
        # for gen_id, new_p in prod_p:
        #     self.numerical_circuit.generator_data.p[gen_id] = new_p  # in MW, I hope
        #
        # # generators' voltage
        # for gen_id, new_v in prod_v:
        #     # vset comes in kV, we need the p.u. set point
        #     vnom = self.generator_bus_nominal_volatges[gen_id]
        #     self.numerical_circuit.generator_data.v[gen_id] = new_v / vnom

        # batteries
        batteries_bus = backendAction.get_storages_bus()
        for i, (changed, p,
                bus_changed, bus_value) in enumerate(zip(storage.changed, storage.values,
                                                         batteries_bus.changed, batteries_bus.values)):
            if changed:
                self.numerical_circuit.battery_data.p[i] = p  # in MW, I hope

            if bus_changed:
                self.numerical_circuit.battery_data.active[i] = int(bus_value != -1)

        # loads
        loads_bus = backendAction.get_loads_bus()
        for i, (changed_p, p,
                changed_q, q,
                bus_changed, bus_value) in enumerate(zip(load_p.changed, load_p.values,
                                                         load_q.changed, load_q.values,
                                                         loads_bus.changed, loads_bus.values)):

            if changed_p and changed_p:
                self.numerical_circuit.load_data.S[i] = complex(p, q)

            elif changed_p and not changed_q:
                Q = self.numerical_circuit.load_data.S[i].imag
                self.numerical_circuit.load_data.S[i] = complex(p, Q)

            elif changed_q and not changed_p:
                P = self.numerical_circuit.load_data.S[i].real
                self.numerical_circuit.load_data.S[i] = complex(P, q)

            else:
                pass  # no changes

            if bus_changed:
                self.numerical_circuit.load_data.active[i] = int(bus_value != -1)

        # shunts
        if shunts__ is not None:
            (shunt_p, shunt_q, shunt_bus) = shunts__
            for i, (changed_p, p,
                    changed_q, q,
                    bus_changed, bus_value) in enumerate(zip(shunt_p.changed, shunt_p.values,
                                                             shunt_q.changed, shunt_q.values,
                                                             shunt_bus.changed, shunt_bus.values)):
                if changed_p and changed_p:
                    self.numerical_circuit.shunt_data.admittance[i] = complex(p, q)

                elif changed_p and not changed_q:
                    Q = self.numerical_circuit.shunt_data.admittance[i].imag
                    self.numerical_circuit.shunt_data.admittance[i] = complex(p, Q)

                elif changed_q and not changed_p:
                    P = self.numerical_circuit.shunt_data.admittance[i].real
                    self.numerical_circuit.shunt_data.admittance[i] = complex(P, q)

                else:
                    pass  # no changes

                if bus_changed:
                    self.numerical_circuit.shunt_data.active[i] = int(bus_value != -1)

        # TODO: what about topology? => Ben: topo information is given in topo__
        # have a look at https://github.com/rte-france/Grid2Op/blob/master/examples/backend_integration/Step4_modify_line_status.py
        # and https://github.com/rte-france/Grid2Op/blob/master/examples/backend_integration/Step5_modify_topology.py

        # handle the disconnection of lines
        lines_or_bus = backendAction.get_lines_or_bus()
        lines_ex_bus = backendAction.get_lines_ex_bus()
        for k, (or_bus_changed, or_bus,
                ex_bus_changed, ex_bus) in enumerate(zip(lines_or_bus.changed, lines_or_bus.values,
                                                         lines_ex_bus.changed, lines_ex_bus.values)):

            if or_bus_changed or ex_bus_changed:
                if or_bus == -1 or ex_bus == -1:
                    self.numerical_circuit.branch_data.active[k] = 0
                else:
                    self.numerical_circuit.branch_data.active[k] = 1

        # reset numerical circuit after the changes
        self.numerical_circuit.reset_calculations()

    def runpf(self, is_dc=False):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Run a power flow on the underlying _grid. This implements an optimization of the powerflow
        computation: if the number of
        buses has not changed between two calls, the previous results are re used. This speeds up the computation
        in case of "do nothing" action applied.
        """

        # power flow options
        pf_options = sim.PowerFlowOptions(max_iter=self.max_iter,
                                          distributed_slack=self.distributed_slack,
                                          solver_type=SolverType.DC if is_dc else SolverType.NR,
                                          retry_with_other_methods=True)
        # run the power flow
        self.results = multi_island_pf_nc(nc=self.numerical_circuit, options=pf_options)

        # return the status
        if self.results.converged:
            return True, None
        else:
            return False, Exception("Did not converge, error ".format(self.results.error))

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
        tmp_ = self._grid
        self._grid = None
        res = copy.deepcopy(self)
        res._grid = tmp_.copy()
        self._grid = tmp_

        # res = GridCalBackend()
        # res.numerical_circuit = self.numerical_circuit.copy()
        # res._grid = self._grid  # this can be a refference since all the stuff is done with numerical_circuit
        # res.initiailize_or_update_internals()

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
        FileSave(circuit=self._grid, file_name=full_path).save()

    def get_line_status(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        As all the functions related to powerline,
        """
        return self.numerical_circuit.branch_data.active

    def get_line_flow(self):
        return self.results.Sf.real

    def _disconnect_line(self, id_):
        self.numerical_circuit.branch_data.active[id_] = 0

    def _reconnect_line(self, id_):
        self.numerical_circuit.branch_data.active[id_] = 1

    def get_topo_vect_old(self):
        """
        Compose or get topo_vect
        the size should agree with the rows of type(self).grid_objects_types
        :return:
        """
        # n = self.numerical_circuit.generator_data.nelm
        # n += self.numerical_circuit.load_data.nelm
        # n += self.numerical_circuit.branch_data.nelm * 2
        # n += self.numerical_circuit.battery_data.nelm
        # bus_idx, gen_idx, load_idx, shunt_idx, line_idx, storage_idx
        # declare a list for each bus
        data = {i: list() for i in range(self.numerical_circuit.nbus)}

        # for each monopole structure ...
        for struct in [self.numerical_circuit.generator_data,
                       self.numerical_circuit.load_data,
                       # self.numerical_circuit.shunt_data  # shunts don't count
                       ]:
            for i, bus_idx in enumerate(struct.get_bus_indices()):
                data[bus_idx].append(i)

        # for each dipole structure ...
        for i, (f, t) in enumerate(zip(self.numerical_circuit.branch_data.F, self.numerical_circuit.branch_data.T)):
            data[f].append(i)
            data[t].append(t)

        # repeat for the battery data to have the same indexing ...
        for struct in [self.numerical_circuit.battery_data]:
            for i, bus_idx in enumerate(struct.get_bus_indices()):
                data[bus_idx].append(i)

        # arrange elements in order
        lst2 = list()
        for bus_idx in range(self.numerical_circuit.nbus):
            lst2 += data[bus_idx]

        # gen_buses = self.numerical_circuit.generator_data.C_bus_elm.sum(axis = 1).A1
        # for i, num in enumerate(gen_buses):
        #     if num == 0:
        #         continue

        return np.array(lst2)
    
    def get_topo_vect(self):
        """
        Compose or get topo_vect
        the size should agree with the rows of type(self).grid_objects_types
        :return:
        """
        topo_list = []
        # n = self.numerical_circuit.generator_data.nelm
        # n += self.numerical_circuit.load_data.nelm
        # n += self.numerical_circuit.branch_data.nelm * 2
        # n += self.numerical_circuit.battery_data.nelm
        # bus_idx, gen_idx, load_idx, shunt_idx, line_idx, storage_idx
        # declare a list for each bus
        data = {i: list() for i in range(self.numerical_circuit.nbus)}

        # for each monopole structure ...
        for struct in [self.numerical_circuit.generator_data,
                       self.numerical_circuit.load_data,
                       # self.numerical_circuit.shunt_data  # shunts don't count
                       ]:
            for i, bus_idx in enumerate(struct.get_bus_indices()):
                if self.numerical_circuit.bus_data.active[bus_idx]:
                    topo_list.append(self.bus_idx_in_sub[bus_idx])
                else:
                    topo_list.append(-1)

        # for each dipole structure ...
        for i, (f, t) in enumerate(zip(self.numerical_circuit.branch_data.F, self.numerical_circuit.branch_data.T)):
            if self.numerical_circuit.bus_data.active[f]:
                topo_list.append(self.bus_idx_in_sub[f])
            else:
                topo_list.append(-1)
            if self.numerical_circuit.bus_data.active[t]:
                topo_list.append(self.bus_idx_in_sub[t])
            else:
                topo_list.append(-1)
            
        # repeat for the battery data to have the same indexing ...
        for struct in [self.numerical_circuit.battery_data]:
            for i, bus_idx in enumerate(struct.get_bus_indices()):
                if self.numerical_circuit.bus_data.active[bus_idx]:
                    topo_list.append(self.bus_idx_in_sub[bus_idx])
                else:
                    topo_list.append(-1)
                

        return np.array(topo_list)

    def generators_info(self):
        if self.results:
            P = self.numerical_circuit.generator_data.p
            Q = np.abs(self.results.Sbus.imag) * self.numerical_circuit.generator_data.C_bus_elm
            # Vm = np.abs(self.results.voltage * self.numerical_circuit.bus_data.Vnom)
            # V = Vm * self.numerical_circuit.generator_data.C_bus_elm
            V = self.generator_bus_nominal_voltages * self.numerical_circuit.generator_data.v
        else:
            P = np.zeros(self.numerical_circuit.ngen)
            Q = np.zeros(self.numerical_circuit.ngen)
            # V = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.generator_data.C_bus_elm
            V = self.generator_bus_nominal_voltages * self.numerical_circuit.generator_data.v

        return P.copy(), Q.copy(), V.copy()

    def loads_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage * self.numerical_circuit.bus_data.Vnom)
            P = self.numerical_circuit.load_data.S.real
            Q = self.numerical_circuit.load_data.S.imag
            V = Vm * self.numerical_circuit.load_data.C_bus_elm

        else:
            P = np.zeros(self.numerical_circuit.nload)
            Q = np.zeros(self.numerical_circuit.nload)
            V = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.load_data.C_bus_elm

        return P.copy(), Q.copy(), V.copy()

    def lines_or_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage * self.numerical_circuit.bus_data.Vnom)
            P = self.results.Sf.real  # MW
            Q = self.results.Sf.imag  # MVAr
            V = self.numerical_circuit.branch_data.C_branch_bus_f * Vm  # kV
            A = (self.results.Sf / (V * 1.732050808) * 1000.0).real  # flow in Amperes
            np.nan_to_num(A, copy=False)  # replace nan by 0 in-place

        else:
            P = np.zeros(self.numerical_circuit.nbr)  # MW
            Q = np.zeros(self.numerical_circuit.nbr)  # MVAr
            V = self.numerical_circuit.branch_data.C_branch_bus_f * self.numerical_circuit.bus_data.Vnom  # kV
            A = np.zeros(self.numerical_circuit.nbr)

        V *= self.numerical_circuit.branch_data.active

        return P.copy(), Q.copy(), V.copy(), A.copy()

    def lines_ex_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage * self.numerical_circuit.bus_data.Vnom)
            P = self.results.St.real  # MW
            Q = self.results.St.imag  # MVAr
            V = self.numerical_circuit.branch_data.C_branch_bus_t * Vm  # kV
            A = (self.results.St / (V * 1.732050808) * 1000.0).real  # flow in Amperes

        else:
            P = np.zeros(self.numerical_circuit.nbr)  # MW
            Q = np.zeros(self.numerical_circuit.nbr)  # MVAr
            V = self.numerical_circuit.branch_data.C_branch_bus_t * self.numerical_circuit.bus_data.Vnom  # kV
            A = np.zeros(self.numerical_circuit.nbr)

        V *= self.numerical_circuit.branch_data.active

        return P.copy(), Q.copy(), V.copy(), A.copy()

    def shunt_info(self):
        shunts_subid = self.numerical_circuit.shunt_data.get_bus_indices()
        shunts_subid = np.array([self.bus_idx_in_sub[i] for i in shunts_subid])

        if self.results:
            Vm = np.abs(self.results.voltage * self.numerical_circuit.bus_data.Vnom)
            S_shunt = self.results.Sbus.real * self.numerical_circuit.shunt_data.C_bus_elm
            P = S_shunt.real  # MW
            Q = S_shunt.imag  # MVAr
            V = Vm * self.numerical_circuit.shunt_data.C_bus_elm  # kV
            # A = (S_shunt / (V * 1.732050808) * 1000.0).real  # flow in Amperes

        else:
            P = np.zeros(self.numerical_circuit.nshunt)  # MW
            Q = np.zeros(self.numerical_circuit.nshunt)  # MVAr
            V = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.shunt_data.C_bus_elm  # kV
            # A = np.zeros(self.numerical_circuit.nshunt)

        return P.copy(), Q.copy(), V.copy(), shunts_subid

    def storages_info(self):
        if self.results:
            Vm = np.abs(self.results.voltage * self.numerical_circuit.bus_data.Vnom)
            P = self.numerical_circuit.battery_data.p
            Q = np.abs(self.results.Sbus.imag) * self.numerical_circuit.battery_data.C_bus_elm
            V = Vm * self.numerical_circuit.battery_data.C_bus_elm
        else:
            P = np.zeros(self.numerical_circuit.nbatt)
            Q = np.zeros(self.numerical_circuit.nbatt)
            V = self.numerical_circuit.bus_data.Vnom * self.numerical_circuit.battery_data.C_bus_elm

        return P.copy(), Q.copy(), V.copy()

    def sub_from_bus_id(self, bus_id):
        # TODO: what to do here?
        # if bus_id >= self._number_true_line:
        #     return bus_id - self._number_true_line
        # Ben: it's an internal function to pandapower backend mainly. Nothing to do as it's not used.
        return bus_id

    def check_kirchoff(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Check that the powergrid respects kirchhoff's law.
        This function can be called at any moment (after a powerflow has been run)
        to make sure a powergrid is in a consistent state, or to perform
        some tests for example.

        In order to function properly, this method requires that :func:`Backend.shunt_info` and
        :func:`Backend.sub_from_bus_id` are properly defined. Otherwise the results might be wrong, especially
        for reactive values (q_subs and q_bus bellow)

        Returns
        -------
        p_subs ``numpy.ndarray``
            sum of injected active power at each substations (MW)
        q_subs ``numpy.ndarray``
            sum of injected reactive power at each substations (MVAr)
        p_bus ``numpy.ndarray``
            sum of injected active power at each buses. It is given in form of a matrix, with number of substations as
            row, and number of columns equal to the maximum number of buses for a substation (MW)
        q_bus ``numpy.ndarray``
            sum of injected reactive power at each buses. It is given in form of a matrix, with number of substations as
            row, and number of columns equal to the maximum number of buses for a substation (MVAr)
        diff_v_bus: ``numpy.ndarray`` (2d array)
            difference between maximum voltage and minimum voltage (computed for each elements)
            at each bus. It is an array of two dimension:

            - first dimension represents the the substation (between 1 and self.n_sub)
            - second element represents the busbar in the substation (0 or 1 usually)

        """

        p_or, q_or, v_or, *_ = self.lines_or_info()
        p_ex, q_ex, v_ex, *_ = self.lines_ex_info()
        p_gen, q_gen, v_gen = self.generators_info()
        p_load, q_load, v_load = self.loads_info()
        if self.n_storage > 0:
            p_storage, q_storage, v_storage = self.storages_info()

        # fist check the "substation law" : nothing is created at any substation
        p_subs = np.zeros(self.n_sub, dtype=float)
        q_subs = np.zeros(self.n_sub, dtype=float)

        # check for each bus
        p_bus = np.zeros((self.n_sub, 2), dtype=float)
        q_bus = np.zeros((self.n_sub, 2), dtype=float)
        v_bus = (
                np.zeros((self.n_sub, 2, 2), dtype=float) - 1.0
        )  # sub, busbar, [min,max]
        topo_vect = self.get_topo_vect()

        # bellow i'm "forced" to do a loop otherwise, numpy do not compute the "+=" the way I want it to.
        # for example, if two powerlines are such that line_or_to_subid is equal (eg both connected to substation 0)
        # then numpy do not guarantee that `p_subs[self.line_or_to_subid] += p_or` will add the two "corresponding p_or"
        # TODO this can be vectorized with matrix product, see example in obs.flow_bus_matrix (BaseObervation.py)
        for i in range(self.n_line):
            sub_or_id = self.line_or_to_subid[i]
            sub_ex_id = self.line_ex_to_subid[i]
            loc_bus_or = topo_vect[self.line_or_pos_topo_vect[i]] - 1
            loc_bus_ex = topo_vect[self.line_ex_pos_topo_vect[i]] - 1

            # for substations
            p_subs[sub_or_id] += p_or[i]
            p_subs[sub_ex_id] += p_ex[i]

            q_subs[sub_or_id] += q_or[i]
            q_subs[sub_ex_id] += q_ex[i]

            # for bus
            p_bus[sub_or_id, loc_bus_or] += p_or[i]
            q_bus[sub_or_id, loc_bus_or] += q_or[i]

            p_bus[sub_ex_id, loc_bus_ex] += p_ex[i]
            q_bus[sub_ex_id, loc_bus_ex] += q_ex[i]

            # fill the min / max voltage per bus (initialization)
            if (v_bus[sub_or_id, loc_bus_or,][0] == -1):
                v_bus[sub_or_id, loc_bus_or,][0] = v_or[i]
            if (v_bus[sub_ex_id, loc_bus_ex,][0] == -1):
                v_bus[sub_ex_id, loc_bus_ex,][0] = v_ex[i]
            if (v_bus[sub_or_id, loc_bus_or,][1] == -1):
                v_bus[sub_or_id, loc_bus_or,][1] = v_or[i]
            if (v_bus[sub_ex_id, loc_bus_ex,][1] == -1):
                v_bus[sub_ex_id, loc_bus_ex,][1] = v_ex[i]

            # now compute the correct stuff
            if v_or[i] > 0.0:
                # line is connected
                v_bus[sub_or_id, loc_bus_or,][0] = min(v_bus[sub_or_id, loc_bus_or,][0], v_or[i], )
                v_bus[sub_or_id, loc_bus_or,][1] = max(v_bus[sub_or_id, loc_bus_or,][1], v_or[i], )

            if v_ex[i] > 0:
                # line is connected
                v_bus[sub_ex_id, loc_bus_ex,][0] = min(v_bus[sub_ex_id, loc_bus_ex,][0], v_ex[i], )
                v_bus[sub_ex_id, loc_bus_ex,][1] = max(v_bus[sub_ex_id, loc_bus_ex,][1], v_ex[i], )

        for i in range(self.n_gen):
            # for substations
            p_subs[self.gen_to_subid[i]] -= p_gen[i]
            q_subs[self.gen_to_subid[i]] -= q_gen[i]

            # for bus
            p_bus[
                self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1
            ] -= p_gen[i]
            q_bus[
                self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1
            ] -= q_gen[i]

            # compute max and min values
            if v_gen[i]:
                # but only if gen is connected
                v_bus[self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1][
                    0
                ] = min(
                    v_bus[
                        self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1
                    ][0],
                    v_gen[i],
                )
                v_bus[self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1][
                    1
                ] = max(
                    v_bus[
                        self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1
                    ][1],
                    v_gen[i],
                )

        for i in range(self.n_load):
            # for substations
            p_subs[self.load_to_subid[i]] += p_load[i]
            q_subs[self.load_to_subid[i]] += q_load[i]

            # for buses
            p_bus[
                self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1
            ] += p_load[i]
            q_bus[
                self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1
            ] += q_load[i]

            # compute max and min values
            if v_load[i]:
                # but only if load is connected
                v_bus[self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1][
                    0
                ] = min(
                    v_bus[
                        self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1
                    ][0],
                    v_load[i],
                )
                v_bus[self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1][
                    1
                ] = max(
                    v_bus[
                        self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1
                    ][1],
                    v_load[i],
                )

        for i in range(self.n_storage):
            p_subs[self.storage_to_subid[i]] += p_storage[i]
            q_subs[self.storage_to_subid[i]] += q_storage[i]
            p_bus[self.storage_to_subid[i], topo_vect[self.storage_pos_topo_vect[i]] - 1] += p_storage[i]
            q_bus[self.storage_to_subid[i], topo_vect[self.storage_pos_topo_vect[i]] - 1] += q_storage[i]

            # compute max and min values
            if v_storage[i] > 0:
                # the storage unit is connected
                v_bus[self.storage_to_subid[i], topo_vect[self.storage_pos_topo_vect[i]] - 1][0] = min(v_bus[self.storage_to_subid[i], topo_vect[self.storage_pos_topo_vect[i]] - 1][0], v_storage[i])
                v_bus[self.storage_to_subid[i], topo_vect[self.storage_pos_topo_vect[i]] - 1][1] = max(v_bus[self.storage_to_subid[i], topo_vect[self.storage_pos_topo_vect[i]] - 1][1], v_storage[i])

        if type(self).shunts_data_available:
            p_s, q_s, v_s, bus_s = self.shunt_info()
            for i in range(self.n_shunt):
                # for substations
                p_subs[self.shunt_to_subid[i]] += p_s[i]
                q_subs[self.shunt_to_subid[i]] += q_s[i]

                # for buses
                p_bus[self.shunt_to_subid[i], bus_s[i] - 1] += p_s[i]
                q_bus[self.shunt_to_subid[i], bus_s[i] - 1] += q_s[i]

                # compute max and min values
                v_bus[self.shunt_to_subid[i], bus_s[i] - 1][0] = min(v_bus[self.shunt_to_subid[i], bus_s[i] - 1][0], v_s[i])
                v_bus[self.shunt_to_subid[i], bus_s[i] - 1][1] = max(v_bus[self.shunt_to_subid[i], bus_s[i] - 1][1], v_s[i])
        else:
            warnings.warn(
                "Backend.check_kirchoff Impossible to get shunt information. Reactive information might be "
                "incorrect."
            )

        diff_v_bus = np.zeros((self.n_sub, 2), dtype=float)
        diff_v_bus[:, :] = v_bus[:, :, 1] - v_bus[:, :, 0]

        # SanPen calcs -------------------------------------------------------------------------------------------------
        dS = self.results.Sbus - self.numerical_circuit.Sbus * self.numerical_circuit.Sbase
        dP = dS.real  # p_subs
        dQ = dS.imag  # q_subs

        # nodal mismatches (numerical method)
        f_err = np.r_[dP[self.numerical_circuit.pqpv], dQ[self.numerical_circuit.pq]]

        err = np.max(np.abs(f_err))
        # --------------------------------------------------------------------------------------------------------------

        return p_subs, q_subs, p_bus, q_bus, diff_v_bus
