import os, sys
import numpy as np
from collections import namedtuple
from openbabel import OBAtom, OBElementTable
import openbabel

Atom = namedtuple('Atom',
                  ['smina_name', 'adname', 'ad_radius', 'ad_depth', 'ad_solvation', 'ad_volume', 'covalent_radius',
                   'xs_radius', 'xs_hydrophobe', 'xs_donor', 'xs_acceptor', 'ad_heteroatom'])

Atom_dict = [
    Atom(*["Hydrogen", "H", 1.000000, 0.020000, 0.000510, 0.000000, 0.370000, 0.000000, False, False, False, False]),
    Atom(*["PolarHydrogen", "HD", 1.000000, 0.020000, 0.000510, 0.000000, 0.370000, 0.000000, False, False, False,
           False]),
    Atom(
        *["AliphaticCarbonXSHydrophobe", "C", 2.000000, 0.150000, -0.001430, 33.510300, 0.770000, 1.900000, True, False,
          False, False]),
    Atom(*["AliphaticCarbonXSNonHydrophobe", "C", 2.000000, 0.150000, -0.001430, 33.510300, 0.770000, 1.900000, False,
           False, False, False]),
    Atom(*["AromaticCarbonXSHydrophobe", "A", 2.000000, 0.150000, -0.000520, 33.510300, 0.770000, 1.900000, True, False,
           False, False]),
    Atom(*["AromaticCarbonXSNonHydrophobe", "A", 2.000000, 0.150000, -0.000520, 33.510300, 0.770000, 1.900000, False,
           False, False, False]),
    Atom(*["Nitrogen", "N", 1.750000, 0.160000, -0.001620, 22.449300, 0.750000, 1.800000, False, False, False, True]),
    Atom(*["NitrogenXSDonor", "N", 1.750000, 0.160000, -0.001620, 22.449300, 0.750000, 1.800000, False, True, False,
           True]),
    Atom(*["NitrogenXSDonorAcceptor", "NA", 1.750000, 0.160000, -0.001620, 22.449300, 0.750000, 1.800000, False, True,
           True, True]),
    Atom(*["NitrogenXSAcceptor", "NA", 1.750000, 0.160000, -0.001620, 22.449300, 0.750000, 1.800000, False, False, True,
           True]),
    Atom(*["Oxygen", "O", 1.600000, 0.200000, -0.002510, 17.157300, 0.730000, 1.700000, False, False, False, True]),
    Atom(*["OxygenXSDonor", "O", 1.600000, 0.200000, -0.002510, 17.157300, 0.730000, 1.700000, False, True, False,
           True]),
    Atom(*["OxygenXSDonorAcceptor", "OA", 1.600000, 0.200000, -0.002510, 17.157300, 0.730000, 1.700000, False, True,
           True, True]),
    Atom(*["OxygenXSAcceptor", "OA", 1.600000, 0.200000, -0.002510, 17.157300, 0.730000, 1.700000, False, False, True,
           True]),
    Atom(*["Sulfur", "S", 2.000000, 0.200000, -0.002140, 33.510300, 1.020000, 2.000000, False, False, False, True]),
    Atom(*["SulfurAcceptor", "SA", 2.000000, 0.200000, -0.002140, 33.510300, 1.020000, 2.000000, False, False, False,
           True]),
    Atom(*["Phosphorus", "P", 2.100000, 0.200000, -0.001100, 38.792400, 1.060000, 2.100000, False, False, False, True]),
    Atom(*["Fluorine", "F", 1.545000, 0.080000, -0.001100, 15.448000, 0.710000, 1.500000, True, False, False, True]),
    Atom(*["Chlorine", "Cl", 2.045000, 0.276000, -0.001100, 35.823500, 0.990000, 1.800000, True, False, False, True]),
    Atom(*["Bromine", "Br", 2.165000, 0.389000, -0.001100, 42.566100, 1.140000, 2.000000, True, False, False, True]),
    Atom(*["Iodine", "I", 2.360000, 0.550000, -0.001100, 55.058500, 1.330000, 2.200000, True, False, False, True]),
    Atom(*["Magnesium", "Mg", 0.650000, 0.875000, -0.001100, 1.560000, 1.300000, 1.200000, False, True, False, True]),
    Atom(*["Manganese", "Mn", 0.650000, 0.875000, -0.001100, 2.140000, 1.390000, 1.200000, False, True, False, True]),
    Atom(*["Zinc", "Zn", 0.740000, 0.550000, -0.001100, 1.700000, 1.310000, 1.200000, False, True, False, True]),
    Atom(*["Calcium", "Ca", 0.990000, 0.550000, -0.001100, 2.770000, 1.740000, 1.200000, False, True, False, True]),
    Atom(*["Iron", "Fe", 0.650000, 0.010000, -0.001100, 1.840000, 1.250000, 1.200000, False, True, False, True]),
    Atom(*["GenericMetal", "M", 1.200000, 0.000000, -0.001100, 22.449300, 1.750000, 1.200000, False, True, False, True])
]

Mental = ["Cu", "Fe", "Na", "k", "Hg", "Co", "U", "Cd", "Ni"]
etab = OBElementTable()


class atom_parser:
    def __init__(self, obatom):
        self.coords = np.array([obatom.x(), obatom.y(), obatom.z()])
        self.charge = obatom.GetPartialCharge()
        self.atom_id = self.parse_atom(obatom)

    def getAtomIdxByName(self, element_name):
        '''
        Some Atom have same element name, their type will be adjusted later
        :param element_name: string
        :return: namedtuple Atom
        '''
        for idx, atom in enumerate(Atom_dict):
            if atom.adname == element_name:
                return idx
        raise Exception('Atom name {} can not find in collection'.element_name)

    def type_shift(self, origin_idx, suffix):
        '''
        Get idx of same atom but different attribute( hydrophobe, doner, acceptor ect.)
        :param origin_idx: int
        :param suffix: string
        :return: idx: int
        '''
        adname = Atom_dict[origin_idx].adname
        for idx, atom in enumerate(Atom_dict):
            if atom.adname == adname and atom.smina_name.endswith(suffix):
                return idx
        raise Exception("Cannot find {} {}.".format(suffix, adname))

    def adjust_info(self, obatom):
        '''
        If a atom is bonded to polarhydrogen or heteroatom
        :param obatom: openbabel OBAtom istance
        :return: bond_2_hd, bond_2_hetero : boolean
        '''
        bonds = list(openbabel.OBAtomBondIter(obatom))
        atom_id = obatom.GetId()

        bond_2_hd = False
        bond_2_hetero = False
        for bond in bonds:
            begin = bond.GetBeginAtom().GetId()
            end = bond.GetEndAtom().GetId()
            if atom_id == begin:
                link_atom = bond.GetEndAtom()
            elif atom_id == end:
                link_atom = bond.GetBeginAtom()
            else:
                raise Exception("Wrong bond")

            if link_atom.IsHydrogen():
                # all the remained hydrogen is polarhydrogen
                bond_2_hd = True
            if link_atom.IsHeteroatom():
                bond_2_hetero = True

        return bond_2_hd, bond_2_hetero

    def parse_atom(self, obatom):
        # parse basic atom type from obatom
        element_name = etab.GetSymbol(obatom.GetAtomicNum());
        if obatom.IsHydrogen():
            element_name = 'HD'
        elif obatom.IsCarbon() and obatom.IsAromatic():
            element_name = 'A'
        elif obatom.IsOxygen():
            element_name = 'OA'
        elif obatom.IsNitrogen() and obatom.IsHbondAcceptor():
            element_name = 'NA'
        elif obatom.IsSulfur() and obatom.IsHbondAcceptor():
            element_name = 'SA'

        if element_name in Mental:
            element_name = 'M'

        idx = self.getAtomIdxByName(element_name)

        # typ shift if a atom is bonded to specific type of atom
        bond_2_hd, bond_2_hetero = self.adjust_info(obatom)

        if element_name == 'A' or element_name == 'C':
            if bond_2_hetero:
                idx = self.type_shift(idx, 'XSNonHydrophobe')
            else:
                idx = self.type_shift(idx, 'XSHydrophobe')
        elif element_name == 'N':
            if bond_2_hd:
                idx = self.type_shift(idx, 'XSDonor')
            else:
                idx = self.type_shift(idx, 'Nitrogen')
        elif element_name == 'NA':
            if bond_2_hd:
                idx = self.type_shift(idx, 'XSDonorAcceptor')
            else:
                idx = self.type_shift(idx, 'XSAcceptor')
        elif element_name == 'O':
            if bond_2_hd:
                idx = self.type_shift(idx, 'XSDonor')
            else:
                idx = self.type_shift(idx, 'Oxygen')
        elif element_name == 'OA':
            if bond_2_hd:
                idx = self.type_shift(idx, 'XSDonorAcceptor')
            else:
                idx = self.type_shift(idx, 'XSAcceptor')

        return idx

    def get_atom(self):
        return Atom_dict[self.atom_id]

