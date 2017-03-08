import os, sys
import numpy as np
from openbabel import OBAtom, OBElementTable
import openbabel
from atom_parse import Atom_dict
from atom_parse import atom_parser as Atom
from collections import namedtuple
from functools import partial

class evaluation:
    const_v = 1000
    const_cap = 100
    const_cutoff = 8
    const_smooth = 1
    const_epsilon = 2.22045e-16

    def __init__(self, receptor, ligand, debug, log):
        self.debug = debug
        self.log = log

        # parse receptor and ligand
        self.obabel_load(receptor, ligand)
        # create scoring function
        self.create_scoring_functions()
        # set transform parameter
        self.set_transform()


    def debug_info(self, message):
        if self.debug == 'print':
            print message
        elif self.debug == 'log':
            with open(self.log, 'a') as fout:
                fout.write(message + '\n')
        elif self.debug == 'off':
            pass

    def obabel_load(self, receptor, ligand):
        '''
        load ligand and receptor through openbabel
        '''
        obConversion = openbabel.OBConversion()
        OBligand = openbabel.OBMol()

        self.debug_info('Parse {} by openbabel.'.format(ligand))

        if not obConversion.ReadFile(OBligand, ligand):
            message = 'Cannot parse {}'.format(ligand)
            raise Exception(message)

        OBligand.DeleteNonPolarHydrogens()
        OBligand.AddPolarHydrogens()
        # lig_atoms = [obatom for obatom in openbabel.OBMolAtomIter(OBligand)]
        self.lig = []
        for obatom in openbabel.OBMolAtomIter(OBligand):
            atom = Atom(obatom)
            self.lig.append(atom)

        OBreceptor = openbabel.OBMol()
        self.debug_info('Parse {} by opnbabel.'.format(receptor))
        if not obConversion.ReadFile(OBreceptor, receptor):
            message = 'Cannot parse {}.'.format(receptor)
            raise Exception(message)

        OBreceptor.DeleteNonPolarHydrogens()
        OBreceptor.AddPolarHydrogens()
        # rec_atoms = [obatom for obatom in openbabel.OBMolAtomIter(OBreceptor)]
        self.rec = []
        for obatom in openbabel.OBMolAtomIter(OBreceptor):
            atom = Atom(obatom)
            self.rec.append(atom)

    def opt_distance(self, atom_a, atom_b):

        return atom_a.get_atom().xs_radius + atom_b.get_atom().xs_radius

    def curl(self, e):
        if e > 0:
            tmp = 1.0 * self.const_v / (self.const_v + e)
            e *= tmp
        return e

    def not_hydrogen(self, atom):
        return not atom.smina_name.endswith('Hydrogen')

    def h_bond_possible(self, atom_a, atom_b):

        return (atom_a.get_atom().xs_donor and atom_b.get_atom().xs_acceptor) or (
            atom_a.get_atom().xs_acceptor and atom_b.get_atom().xs_donor)

    def anti_h_bond(self, atom_a, atom_b):
        if (atom_a.get_atom().xs_donor and not atom_a.get_atom().xs_acceptor):
            return atom_b.get_atom().xs_donor and not atom_b.get_atom().xs_acceptor

        if (not atom_a.get_atom().xs_donor and atom_a.get_atom().xs_acceptor):
            return not atom_b.get_atom().xs_donor and atom_b.get_atom().xs_acceptor

        return False

    def slope_step(self, surface_distance, bad, good):

        if bad < good:
            if surface_distance <= bad:
                return 0
            if surface_distance >= good:
                return 1
        else:
            if surface_distance >= bad:
                return 0
            if surface_distance <= good:
                return 1
        return 1.0 * (surface_distance - bad) / (good - bad)

    def guass(self, atom_a, atom_b, distance, o=3., w=2.):
        opt_distance = self.opt_distance(atom_a, atom_b)
        surface_distance = distance - opt_distance
        e = np.exp(-np.power((surface_distance - o) / w, 2))
        # print e
        return e

    def vdw(self, atom_a, atom_b, distance, m=12, n=6):
        opt_distance = self.opt_distance(atom_a, atom_b)
        c_i = np.power(opt_distance, n) * m / (n - m)
        c_j = np.power(opt_distance, m) * n / (m - n)

        if distance > opt_distance + self.const_smooth:
            distance -= self.const_smooth
        elif distance < opt_distance - self.const_smooth:
            distance += self.const_smooth
        else:
            distance = opt_distance

        r_i = np.power(distance, n)
        r_j = np.power(distance, m)

        if (r_i > self.const_epsilon or r_j > self.const_epsilon):
            return min(self.const_cap, c_i / r_i + c_j / r_j)
        else:
            return self.const_cap

    def non_dir_h_bond_lj(self, atom_a, atom_b, distance, offset):
        if self.h_bond_possible(atom_a, atom_b):
            d0 = offset + self.opt_distance(atom_a, atom_b)
            n = 10
            m = 12
            depth = 5
            c_i = np.power(d0, n) * depth * m / (n - m)
            c_j = np.power(d0, m) * depth * n / (m - n)

            r_i = np.power(distance, n)
            r_j = np.power(distance, m)

            if (r_i > self.const_epsilon or r_j > self.const_epsilon):
                return min(self.const_cap, c_i / r_i + c_j / r_j)
            else:
                return self.const_cap
        else:
            return 0

    def replusion(self, atom_a, atom_b, distance, offset=0.):
        diff = distance - offset - self.opt_distance(atom_a, atom_b)
        return np.power(diff, 2) if diff < 0 else 0

    def hydrophobic(self, atom_a, atom_b, distance, good, bad):
        if atom_a.get_atom().xs_hydrophobe and atom_b.get_atom().xs_hydrophobe:
            surface_distance = distance - self.opt_distance(atom_a, atom_b)
            return self.slope_step(surface_distance, bad=bad, good=good)
        else:
            return 0

    def non_hydrophobic(self, atom_a, atom_b, distance, good, bad):
        if not atom_a.get_atom().xs_hydrophobe and not atom_b.get_atom().xs_hydrophobe:
            surface_distance = distance - self.opt_distance(atom_a, atom_b)
            return self.slope_step(surface_distance, bad=bad, good=good)
        else:
            return 0

    def non_dir_h_bond(self, atom_a, atom_b, distance, good, bad):
        if self.h_bond_possible(atom_a, atom_b):
            surface_distance = distance - self.opt_distance(atom_a, atom_b)
            return self.slope_step(surface_distance, bad=bad, good=good)
        else:
            return 0

    def non_dir_anti_h_bond_quadratic(self, atom_a, atom_b, distance, offset):
        if self.anti_h_bond(atom_a, atom_b):
            surface_distance = distance - offset - self.opt_distance(atom_a, atom_b)
            if surface_distance > 0:
                return 0
            return surface_distance * surface_distance
        return 0

    def eval(self, atom_a, atom_b, distance):
        e = self.vdw(atom_a, atom_b, distance)
        return e

    def eval_intra(self):
        '''
        We assume ligand is rigid, so intra molecular energy doesn't change we .
        Don't need to calculate it yet.
        :return:
        '''
        distance = lambda i, j: np.sqrt(np.sum(np.power(self.lig[i].coords - self.lig[j].coords, 2)))

        intra_pairs = [(i, j) for i in range(len(self.lig) - 1) for j in range(i + 1, len(self.lig))]
        eval_pairs = [(self.lig[i], self.lig[j], distance(i, j)) for (i, j) in intra_pairs if
                      distance(i, j) < self.const_cutoff]

        energy = 0
        for scoring_term in self.scoring_functions:
            self.debug_info("calculating {} ...".format(scoring_term.name))
            this_e = np.sum(map(lambda (a, b, d): scoring_term.func(a, b, d), eval_pairs))
            self.debug_info("{} original value {}.".format(scoring_term.name, this_e))
            weighted_energy = scoring_term.weight * this_e
            energy += self.curl(weighted_energy)

        return energy

    def eval_inter(self):
        '''
        eval the intermolecular energy
        :return: energy: float weighted score
        '''
        distance = lambda i, j: np.sqrt(np.sum(np.power(self.transform(self.lig[i].coords) - self.rec[j].coords, 2)))

        # doesn't count hydrogen in receptor and lignad
        inter_pairs = [(i, j) for i in range(len(self.lig)) for j in range(len(self.rec)) if
                       self.not_hydrogen(self.lig[i].get_atom()) and self.not_hydrogen(self.rec[j].get_atom())]

        # filter atom pair which is too far away
        eval_pairs = [(self.lig[i], self.rec[j], distance(i, j)) for (i, j) in inter_pairs if
                      distance(i, j) < self.const_cutoff]

        energy = 0
        for scoring_term in self.scoring_function:
            self.debug_info("calculating {} ...".format(scoring_term.name))
            this_e = np.sum(map(lambda (a, b, d): scoring_term.func(a, b, d), eval_pairs))
            self.debug_info("{} original value {}.".format(scoring_term.name, this_e))
            weighted_energy = scoring_term.weight * this_e
            energy += self.curl(weighted_energy)

        return energy

    def set_transform(self, shift=[0, 0, 0], rotate=[0, 0, 0]):
        self.shift = shift
        self.rotate = rotate

    def transform(self, coords):
        # only simple shift now
        return self.shift + coords

    def eval(self):
        '''
         We just eval intermolecular energy
        :return: float, curled energy
        '''

        e = self.eval_inter()
        self.debug_info('intermolecular energy {}'.format(e))
        return e

    def smina_default(self):
        '''
        Create scoring function as defualt smina setting
        :return:
        '''
        scoring_term = namedtuple('scoring_term', ['name', 'weight', 'func'])
        self.scoring_function = []
        self.scoring_function.append(scoring_term('guass_0_0.5', -0.035579, partial(self.guass, o=0, w=0.5)))
        self.scoring_function.append(scoring_term('guass_3_2', -0.005156, partial(self.guass, o=3., w=2. )))
        self.scoring_function.append(scoring_term('replusion_0', 0.840245, partial(self.replusion, o=0.)))
        self.scoring_function.append(scoring_term('hydrophobic_0.5_1.5', -0.035069, partial(self.hydrophobic, good=0.5, bad=1.5)))
        self.scoring_function.append(scoring_term('non_dir_h_bond_-0.7_0', -0.587439, partial(self.non_dir_h_bond, good=-0.7, bad=0.)))

    def create_scoring_functions(self):
        '''
        Combine different scoring terms as final scoring function and assign different weight to each of them.
        :return:
        '''

        self.smina_default()

        '''
        All scoring term available:

        scoring_term = namedtuple('scoring_term', ['name', 'weight', 'func'])
        self.scoring_function = []

        self.scoring_function.append(scoring_term('vdw_12_6', 1.0, partial(self.vdw, m=12, n=6)))
        self.scoring_function.append(scoring_term('guass_3_2', 1.0, partial(self.guass, o=3., w=2.)))
        self.scoring_function.append(scoring_term('replusion_0', 1.0, partial(self.replusion, offset=0.)))
        self.scoring_function.append(
            scoring_term('hydrophobic_0.5_1.5', 1.0, partial(self.hydrophobic, good=0.5, bad=1.5)))
        self.scoring_function.append(
            scoring_term('non_dir_h_bond_-0.7_0', 1.0, partial(self.non_dir_h_bond, good=-0.7, bad=0)))
        self.scoring_function.append(scoring_term('non_dir_anti_h_bond_quadratic_1', 1.0,
                                                  partial(self.non_dir_anti_h_bond_quadratic, offset=1.)))
        self.scoring_function.append(
            scoring_term('non_dir_h_bond_lj_-1', 1.0, partial(self.non_dir_h_bond_lj, offset=-1.)))
        '''
