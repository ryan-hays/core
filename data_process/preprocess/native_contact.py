import numpy as np

def make_pair(x, y):
    xx, yy = np.meshgrid(range(len(x)), range(len(y)))
    return np.c_[xx.ravel(), yy.ravel()]


def compute_distance(x, y, pairs):
    return np.array([np.sqrt(np.sum(np.power(x[i] - y[j], 2))) for (i, j) in pairs])


def compute_native_contact(x, y, pairs, maximum, minimum):
    distance = compute_distance(x, y, pairs)

    distance[distance > maximum] = minimum

    Numerator = np.sum(distance > minimum)

    return Numerator * 1.0 / len(pairs)


def native_contact(receptor, native, ligands):
    '''
        input:
        receptor :
                    numpy array in shape of  (n_r,3) , n_r is the number of atoms
        native :
                    numpy array in shape of  (n_n,3) , n_n is the number of atoms
        ligands:
                    numpy array in shape of  (m,n_l,3) , m is the number of ligands , n_l is the number of atoms

        output:
        native_contact_value :
                    numpy array in shape of (m)
    '''

    maximum_distance = 4.5
    minimum_distance = 3

    pairs = make_pair(receptor, native)
    pairs_distance = compute_distance(receptor, native, pairs)
    pairs_distance[pairs_distance > maximum_distance] = minimum_distance
    native_contact_pair = pairs[pairs_distance > minimum_distance]
    native_contacts = np.array(
        [compute_native_contact(receptor, ligand, native_contact_pair, maximum_distance, minimum_distance) for ligand in ligands])
    print native_contacts
    return native_contacts