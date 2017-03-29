# TARGET

The pdb id and ligand id recored under this foldr is coming from [PDB](http://www.rcsb.org/).
[Advanced Search](http://www.rcsb.org/pdb/search/advSearch.do?search=new) allowed user to submit query with multiple criteria.

The criteria we use are:
- Has Ligand(s)
- Wild Type Protein
- Binding Affinity
- X-ray Resolution



##  Search by KI
### Search criteria
    - Has Lignad(s) : yes
    - Wild Type Protein :
        - Include Expression Tags Yes
        - Percent coverage of UniProt sequence Any
    - Binidng Affinity :
        - Affinity Type : Ki
        - Binidng Affinity : No Limit
    - X-ray Resolution:
        - Between 0 and 2.5

### result


    ```bash
Search Parameter:
Ligand Search : Has free ligands=yes and Percent Sequence Alignment Search : PDB can contain Expression Tag sequence = Yes , and Binding Affinity: Affinity Type is Ki and Resolution is between 0.0 and 2.5

Your search is an ENTITY-BASED QUERY - the 4,331 entities (unique chains) map to 3,753 PDB entries (structures).
    ```


    - Structure :
        - number : 3753
        - file : Ki_structures.txt , one line split by ', '
    - Ligands :
        - number : 3133
        - file : Ki_ligands.txt , one line split by ' '


## Search by IC50
### Search criteria
    - Has Lignad(s) : yes
    - Wild Type Protein :
        - Include Expression Tags Yes
        - Percent coverage of UniProt sequence Any
    - Binidng Affinity :
        - Affinity Type : IC50
        - Binidng Affinity : No Limit
    - X-ray Resolution:
        - Between 0 and 2.5

### Result

```bash
Search Parameter:
Ligand Search : Has free ligands=yes and Percent Sequence Alignment Search : PDB can contain Expression Tag sequence = Yes , and Binding Affinity: Affinity Type is IC50 and Resolution is between 0.0 and 2.5

Your search is an ENTITY-BASED QUERY - the 4,895 entities (unique chains) map to 4,363 PDB entries (structures).
```
    - Structure :
        - number : 4363
        - file : IC50_structures.txt , one line split by ', '
    - Ligands:
        - number : 3952
        - file : IC50_ligands.txt , one line splt by ' '




