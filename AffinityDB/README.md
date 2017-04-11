# AffinityDB
This module is used to prepare data and collect information for them.

## requirement
- openbabel
    
## usage
### Setting parameters in config.py

- **database_root** : the folder to store everyting generated by this module.
- **smina** : the path for smina executable file

### database create option

```bash
python database_create.py --[option]
```

**Option:**
- download
- split
- rotbond
- similarity
- vinardo_dock
- smina_dock
- cleanempty
- rmsd
- overlap
- native_contact
- addh
- minimize

#### download
Each pdb structure in [Protein Data Bank](http://www.rcsb.org/pdb/home/home.do) has an unique id with 4 letter or number like [3EML](http://www.rcsb.org/pdb/explore/explore.do?structureId=3eml). This option get the id of the pdb structure from `config.list_of_PDB_to_download`, and download pdb file to `config.pdb_download_path`.


#### split
The pdb file downloaded from Protein DataBank is co-crystal strucute with proteins, nucleics, residues, water and single meal atoms inside. This option select the receptors and ligands from it, and store them to `config.splited_receptors_path` and `config.splited_ligand_path`.

Receptors here means `protein or nucleic`

Ligands here means `(hetero and not water) or resname ATP or resname ADP or resname AMP or resname GTP or resname GDP or resname GMP`

Splited receptors named by `[pdbid]` e.g. `3eml.pdb`

Splited ligands named by `[pdbid]_[residue_name]_[residue_id]_ligand` e.g. `3eml_SO4_123_ligand.pdb` 


#### rotbond 
**Require Openbabel Python binding**

Count the number of rotable bonds for the ligands. 

#### similarity
The finger print encode molecular structure into binary bits string. We take the similarity between the binary string as the similarity between two ligands.  
This option use OpenBabel to calculate similarity between the ligands splited from same pdb file. 
Available finger print types: 
- FP2
- FP3
- FP4
- MACCS 

```bash
babel -d ligand-a.pdb ligand-b.pdb -ofpt -xfFP2
```

[Tutorial:Fingerprints](https://openbabel.org/wiki/Tutorial:Fingerprints)

#### vinardo_dock
Docking ligands back to their binding pocket by smina, the scoring function is **[vinardo](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155183)**

#### smina_dock
Docking ligands back to their binding pocket by smina, the scoring function is the default scoring function for smina.

#### cleanempty
Check if the docking result is empty and if the docking result can be parsed by prody.
Remove failed docking result.

#### rmsd
Calculate the [RMSD](https://www.wikiwand.com/en/Root-mean-square_deviation_of_atomic_positions) for docking result with the ligands splited from the same pdb and have same residue name. 

#### overlap
Calculate the overlap ratio for docking result among the ligands splited from the same pdb.
Overlap ratio measure how close the _ligand-a_ to the _ligand-b_:
 - _A_ is an atom of the _ligand-a_, if exists atom _B_ of the _ligand-b_ where distance(A,B) <  `config.clash_cufoff_A`, we call A _ovalap atom_
 - overlap ratio of _ligand-a_ = number(overlap atoms of _ligand-a_ ) / number(all atoms of _ligand-a_)

#### native_contact
Calculate native contact ratio for the docking result with the splited ligand.
When docking the splited ligand back to its binding pocket, most time the result will not be docked to exactly the same place.
native contact ratio is an measure about how close the docked result _conf-a_ to the splited ligand _conf-ori_ ( _conf-a_ and _conf-ori_ are different comformation of the same molecule, they have same atoms but different coordinate)
 - _c(A)_  is the coordinate of atom _A_ in _conf-a_, _c'(A)_ is the coordinate of atom _A_ in _conf-ori_, _c''(P)_ is the coordinate of atom _P_ in receptor .
  - native contact ratio with cutoff _D_ = number( distance(_c'(A)_,_c''(P)_) < _D_ and distance(_c(A)_,_c''(P)_) < _D_) / number( distance(_c(A)_,_c''(P)_) < _D_ )


#### addh
Make all hydrogen explicit

```bash
obabel -ipdb input_file -opdb output_file -h
```

Results will be put in correspond folder with suffix `_hydrogens` e.g.:  `3_vinardo_dock` and `3_vinardo_dock_hydrogens`

#### minimize
**Require Openbabel Python binding**

minimize the energy of the molecule (only minimize hydrogen, keep other atoms fixed)

Results will be put in correspond folder with suffix `_minimize` e.g: `4_smina_dock` and `4_smina_dock_minimize`

