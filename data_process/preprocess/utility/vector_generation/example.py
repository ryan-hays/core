'''
Show how to do box generation on the fly

I try to use the least dependency but there still remains some

pls, if this cannot work by default, run :

pip install biopython pyparsing numpy scipy matplotlib prody

in your clean python environment.

'''
from vector import vector_generator,Box

def generate_box_onfly(receptor_filename,ligand_filename,OUT=False,verbose=False):
    '''

    :param receptor_filename:
    :param ligand_filename:
    :param OUT: This is for debug purpose , it will output a complex pdb file with ligand and receptor in cubic box.
    :return: nothing but a vector
    '''

    # Later you can even move box when you want to rotate it.
    # I think someday it will need so I put in backend.
    # For now it will first try to rotate, if failed over 100 times (ligand out of border), then try shift the box.
    whatever = vector_generator(receptor_filename, Boxsize=20, Boxrange=1)

    Tag, Vec= whatever.generate_vector_from_file(ligand_filename, OUT=OUT,verbose=verbose)
    if Tag==False:
        print 'Unfortunately ,sth wrong happened in script!'
    else:
        print 'You will get a result'
        return Vec

if __name__=='__main__':
    rname = 'example.pdb'
    #support mol2 and pdb , I suppose
    lname = 'example_ligand.mol'

    '''
    Here is how to use, very simple, just prepare ligand file and pdb file and you will get the result
    if you want to see pdb results , set OUT=True, otherwise leave it off (simply do not contain OUT=xxx)
    if you want to see verbose print in python , set verbose=True, otherwise leave it off
    '''
    Vec = generate_box_onfly(rname,lname,OUT=True)
    print Vec


    '''
    This is a test module to check if everything goes correctly.
    '''
    #B= Box(center=[12.5,45.36,22.52],Boxsize=20,Boxrange=1)
    #B.transform(rotation=[0.2,0.3,0.5],transition=[0.12,0.05,-0.98])
    #B.self_test()