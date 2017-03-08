## vina score
simple python implement of charge independent scoring in smina

### usage
```bash
$ python vina_score.py -r receptor.pdb -l ligand.pdb
```

### vina_score option
- -r: receptor path
- -l: ligand path
- -d: **print,log,off** print debug information or record in log file or do nothing
- -log: path to write down log information when -d be set as log

### scoring function
Combine of different scoring term like vdw, guass ect.
Every scoring term contain three field: name, weight and func
- name : **string** name of scoring term
- weight: **float** how this scoring term contribute to the final score
- func: **function** some scoring term function need additional input like m,n,offset ect. assgin them here

```python
self.scoring_function.append(scoring_term('vdw_12_6', 1.0, partial(self.vdw, m=12, n=6)))
```

### avaliable scoring term function and additional input
- vdw: m,n
- guass: o,w
- replusion: offset
- hydrophobic: good, bad
- non_hydrophobic: bood, bad
- non_dir_h_bond: good, bad
- non_dir_h_bond_lj: good, bad
- non_dir_anti_h_bond_quadrtic: offset


## visualize
draw graph for scoring result

### usage
```bash
python visualize.py -i energy_map.npy -d images -a X -n 10
```

### option
- -i: input file, the output of vina_score
- -d:(images) dest folder, the perfix of the folder to store generated images
- -a:(X) fixed axis, slice along which axis to generate images
- -n:(10) images num, number of generated images