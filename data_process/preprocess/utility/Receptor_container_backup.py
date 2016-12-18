__author__= 'wy'

#TODO Add comments to make people understand
#TODO This shall be refractored with autodock part so that logic will not be too nasty

import collections
import logging

import numpy as np
import prody as pd

from data_process.preprocess.Config import result_PREFIX,temp_pdb_PREFIX
from data_process.preprocess.utility.autodock_utility import *
from data_process.preprocess.native_contact import native_contact

'''
Core part for generating vectors and split source pdb files with
With the help of prody library.
Need installation first
Using:
  sudo pip install -r requirements.txt

For local installation, see: http://prody.csb.pitt.edu/downloads/

'''

# Tag for hetero atoms
HETERO_PART = 'L'
# Tag for protein atoms
PROTEIN_PART = 'P'

# 8 type of map
electype= ['A','C','d','e','HD','N','NA','OA']


# score endurance with confidence
CONFIDENCE = 0.85

def list_formatter(table):
    '''
    I don't know if there is a better solution to format a list into string
    :param table:
    :return:
    '''
    try:
        output='['+str(table[0])
        for i in range(len(table)-1):
            output+=(','+str(table[i+1]))
        output+=']'
    except:
        raise TypeError('This object is not iterable!')
    return output


class pdb_container:
    '''
    For real pdb-ligand data
    It will separate each ligand (except ions)
    '''
    def get_pdb_type(self):
        '''
        Nucleic Protein or Complex
        :return:
        '''
        if self.pure_protein is not None:
            if self.pure_nucleic is None:
                return 'Protein'
            else:
                return 'Protein_Nucleic_Complex'
        else:
            if self.pure_protein is None:
                return 'Nucleic'
            else:
                return 'Unknown or empty'

    def get_residue_onfly(self,resid):
        '''

        :param resid:
        :return:
        '''
        for pick_one in pd.HierView(self.hetero).iterResidues():
            # less than 3 atoms may be not ok
            if str(pick_one.getResindex())==resid:
                print 'here'
                self.bundle_ligand_data(pick_one,fake_ligand=False,OUT=True)



    def __init__(self,PDB,filepos=None,OUT=True,**kwargs):
        '''

        :param PDB: name of PDB
        :param filepos: directory of where PDB file stores
        :param OUT: if true, splitted files will be output in './data' folder
        :param kwargs: for further extension
        '''
        self.PDBname= PDB
        self.heterodict = {}
        self.ct=0
        self.sequence = {}
        self.pure_protein= None
        self.pure_nucleic= None
        self.pdb_filename = filepos.split('/')[-1]

        if 'BOX' in kwargs:
            self.BOX_range = kwargs['BOX']
        else:
            self.BOX_range = 20
        if 'Size' in kwargs:
            self.BOX_size = kwargs['Size']
        else:
            self.BOX_size = 1

        if filepos is None:
            pdb_store_dir = os.path.join(temp_pdb_PREFIX,PDB)
        else:
            pdb_store_dir = os.path.join(temp_pdb_PREFIX,PDB+''.join(map(lambda xx:(hex(ord(xx))[2:]),os.urandom(16))))
        self.store_dir =pdb_store_dir

        if not os.path.exists(pdb_store_dir):
            os.mkdir(pdb_store_dir)




        # filepos is to determine whether we download pdb files from wwPDB
        # or use what we have
        # Using downloaded is better
        # parse header for first time


        try:
            if filepos is not None:
                parse,header = pd.parsePDB(filepos,header=True)
            else:
                parse,header = pd.parsePDB(PDB,header=True)
                filepos=PDB+'.pdb.gz'
        except:
            #raise IOError
            print filepos
            logging.warning('PDB {} is ignored due to file-not-found error'.format(PDB))
            return
        #Save resolution
        try:
            self.resolution = header['resolution']
        except:
            self.resolution = 'NA'

        #Copy the file


        self.pure_protein = parse.select('protein')
        self.pure_nucleic = parse.select('nucleic')

        # dirty way to throw away nucleic one
        if self.pure_nucleic is not None:
            return
        copy_pdbfile(filepos, pdb_store_dir+'/{0}.pdb'.format(PDB), zipped=filepos.split('.')[-1] == 'gz')

        #repair by guess, i think
        repair_pdbfile(pdb_store_dir+'/{0}.pdb'.format(PDB), PDB)
        #Generating sequence here
        #storage = []
        #split files by chain
        try:
            parse = pd.parsePDB(pdb_store_dir+'/{0}.pdb'.format(PDB))
        except:
            raise IOError('Cannot parse added H')

        self.chain_list= []
        for chain in parse.getHierView():
            #print chain
            #for seq in storage:
            #    if chain.getSequence()==seq:
            #        continue
            self.chain_list.append(chain.getChid())
            self.sequence[chain.getChid()] = chain.getSequence()
            #storage.append(chain.getSequence())

        #now try to fix the pdb from autodock tools

        hetero = parse.select('(hetero and not water) or resname ATP or resname ADP')

        other = parse.select('protein or nucleic')
        self.hetero = hetero
        self.receptor= other

        # print parse.numAtoms(), hetero.numAtoms(), other.numAtoms()

        # if OUT:
        if other is not None:
            pd.writePDB(pdb_store_dir+'/{0}_receptor.pdb'.format(PDB), other)
            #repair_pdbfile('data/{0}/{0}_receptor.pdb'.format(PDB),PDB)
        else:
            return
        # Make vectors for every single hetero parts
        # Their values will be stored in a dict
        #self.register_all_ligand_onsite(hetero,OUT=OUT)

    def register_all_ligand_onsite(self,hetero_part,OUT=True):
        for pick_one in pd.HierView(hetero_part).iterResidues():
           # less than 3 atoms may be not ok
            if pick_one.numAtoms() <= 3:
                continue

            self.bundle_ligand_data(pick_one,fake_ligand=False,OUT=OUT)



    def bundle_ligand_data(self,pick_one,fake_ligand=True,OUT=True,compare_ResId_native='default',Id_suffix='default',filename=None,benchmark=None):
        '''

        :param pick_one:
        :param fake_ligand:
        :param OUT:
        :param compare_ResId_native:
        :param Id_suffix:
        :param filename:
        :param benchmark:
        :return:
        '''
        PDB = self.PDBname
        if fake_ligand==False:
            ResId = str(pick_one.getResindex())
        else:
            ResId = compare_ResId_native + '_' + str(Id_suffix)

        pdb_store_dir = self.store_dir
        other = self.receptor
        # Extract this ligand from protein (as input for openbabel)



        if filename is None:
            filename = pdb_store_dir + '/{1}/{0}_{1}_ligand.pdb'.format(PDB, ResId)
            if not os.path.isfile(filename):
                if not os.path.exists(pdb_store_dir + '/' + ResId):
                    os.mkdir(pdb_store_dir + '/' + ResId)
            if OUT:
                try:
                    pd.writePDB(filename, pick_one)
                    tar_filename = ''.join(filename.split('.')[:-1])
                    tar_filename+='.mol'
                    pdb_to_mol2(filename, tar_filename)
                except:
                    print 'Unexpected Error!'
                    logging.error('Cannot convert {} to mol2 format!'.format(filename.split('/')[-1]))
                    return

        if not os.path.isfile(filename):
            if not os.path.exists(pdb_store_dir + '/' + ResId):
                os.mkdir(pdb_store_dir + '/' + ResId)

        naming = '{}_{}'.format(PDB, ResId)


        # Get coordinate of center
        xyz = pick_one.getCoords()
        middle = pd.calcCenter(pick_one)
        # in pi degree , the rotation of the box (if needed)
        rotation = [0, 0, 0]

        scale = max(max(xyz[:, 0]) - middle[0], middle[0] - min(xyz[:, 0]),
                    max(xyz[:, 1]) - middle[1], middle[1] - min(xyz[:, 1]),
                    max(xyz[:, 2]) - middle[2], middle[2] - min(xyz[:, 2]))



        # assert scale <= 10
        if scale > self.BOX_range / 2:
            logging.warning(
                'Warning! {} has a ligand out of box scale with {} atom distance to center'.format(PDB, scale))
            # Now shifting the boxes:
            max_scale = max(max(xyz[:, 0]) - min(xyz[:, 0]),
                            max(xyz[:, 1]) - min(xyz[:, 1]),
                            max(xyz[:, 2]) - min(xyz[:, 2]))
            if max_scale > self.BOX_range:
                logging.error(
                    'Assertion failed, {} has a ligand out of box completely with scale'.format(PDB, scale))
                return
            # Try to move to the new center
            temp_mid = [(max(xyz[:, 0]) + min(xyz[:, 0])) / 2, (max(xyz[:, 1]) + min(xyz[:, 1])) / 2,
                        (max(xyz[:, 2]) + min(xyz[:, 2])) / 2]

            temp_mid[0] = round(temp_mid[0], 6)
            temp_mid[1] = round(temp_mid[1], 6)
            temp_mid[2] = round(temp_mid[2], 6)
            middle = np.array(temp_mid)
            print middle

        # print middle
        scale_extension = (self.BOX_range - self.BOX_size) / 2
        box_num = int(np.ceil(self.BOX_range / self.BOX_size))
        xx, yy, zz = np.meshgrid(np.linspace(middle[0] - scale_extension, middle[0] + scale_extension, box_num),
                                 np.linspace(middle[1] - scale_extension, middle[1] + scale_extension, box_num),
                                 np.linspace(middle[2] - scale_extension, middle[2] + scale_extension, box_num))

        # print xx
        vector = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        num_vector = [0] * len(vector)

        #print len(vector), box_num
        for atom in pick_one.iterAtoms():
            x, y, z = atom.getCoords()
            x_pos = int(round(x - vector[0][0]))
            # assert 0 <= x_pos <= 19
            y_pos = int(round(y - vector[0][1]))
            # assert 0 <= y_pos <= 19
            z_pos = int(round(z - vector[0][2]))
            # assert 0 <= z_pos <= 19
            if 0 <= x_pos < box_num and 0 <= y_pos < box_num and 0 <= z_pos < box_num:
                # Simply change here to fulfill the mark as 'H_1'
                # note (z(y(x))) follows from atuogrid map file format , otherwise the coordinate system is not correspond coorectly
                num_vector[z_pos * box_num * box_num + y_pos * box_num + x_pos] = atom.getName() + '_' + str(HETERO_PART)

        # quick,dirty way to find atoms of protein in cubic boxes
        pd.defSelectionMacro('inbox',
                          'abs(x-{1}) <= {0} and abs(y-{2}) <= {0} and abs(z-{3}) <= {0}'.format(self.BOX_size / 2,
                                                                                                 middle[0], middle[1],
                                                                                                 middle[2]))
        residues = other.select('protein and same residue as within 18 of center', center=middle)



        if residues is None:
            logging.warning('{} in {} has no atoms nearby'.format(ResId, PDB))
            return


        # This place might have some potential problem
        # for ADP or ATP , they might either be part of nucleic and the ligand
        # This will cause a severe bug when calculating autovina score
        # TODO fix this issue
        nearby = residues.select('inbox')

        if nearby is not None:
            for atom in nearby.iterAtoms():
                x, y, z = atom.getCoords()
                x_pos = int(round(x - vector[0][0]))
                # assert 0 <= x_pos <= 19
                y_pos = int(round(y - vector[0][1]))
                # assert 0 <= y_pos <= 19
                z_pos = int(round(z - vector[0][2]))
                # assert 0 <= z_pos <= 19
                temp = z_pos * box_num * box_num + y_pos * box_num + x_pos
                if 0 <= x_pos < box_num and 0 <= y_pos < box_num and 0 <= z_pos < box_num and num_vector[
                    temp] == 0:
                    # Simply change here to fulfill the mark as 'C_2'
                    num_vector[temp] = atom.getName() + '_' + str(PROTEIN_PART)
                else:
                    # num_vector[temp] += '|'+atom.getName() + '_' + str(PROTEIN_PART)
                    print atom.getName()
                    logging.warning('Coorinate {} {} {} found at {}'.format(x_pos, y_pos, z_pos, self.PDBname))

        # Save into the dict for future locating
        # naming = '{}_{}'.format(PDB, ResId)

        # Do autogrid mapgeneration:
        # ligand_filename = os.path.join(temp_pdb_PREFIX, PDB + '/' + naming + '_ligand.pdb')
        # receptor_filename = os.path.join(temp_pdb_PREFIX, PDB + '/' + naming + '_receptor.pdb')
        # complex_filename = os.path.join(temp_pdb_PREFIX, PDB + '/' + naming + '_complex.pdb')
        # fake_ligand_filename = os.path.join(temp_pdb_PREFIX, 'fake-ligand.pdb')

        self.heterodict[ResId] = {
            'raw_vector': num_vector,
            'center': middle,
            'rotation': rotation,
            'naming': '{}_{}'.format(PDB, ResId),
            'chain': 'NA',
            'filename': filename,
            'id': ResId,
            'Resname': 'NA',
            'ligand': pick_one,
            'protein': residues,
            'vina_score': 'NA',
            'original_one': True,
            'file_generated': False,
            'fake_ligand' : True,
            'RMSF': 0,
            'Contact Similarity': 1,
            'gridmap_protein': 'NA',
            'gridmap_ligand': 'NA',
            'gridmap_complex': 'NA'
        }

        if fake_ligand== True:
            try:
                dist =self._calcRMSD(self.heterodict[compare_ResId_native]['ligand'],pick_one,benchmark=benchmark)
                print dist
                self.heterodict[ResId]['RMSF'] = dist
            except:
                print 'oops'
                raise IOError
            self.heterodict[ResId]['Contact Similarity']= self._calcQ(self.heterodict[compare_ResId_native]['ligand'],
                                                                      pick_one,benchmark=benchmark)
        else:
            self.heterodict[ResId]['Resname']= pick_one.getResname()
            self.heterodict[ResId]['chain'] = pick_one.getChid()

    def _calcRMSD(self,src,tar,benchmark=None):
        '''

        :param src:
        :param tar:
        :param benchmark: very important, to mark tar's order with src with the exactly same coordinate in benchmark
                        but in different order.
        :return:
        '''
        #TODO finish this stuff
        src_heavy = src.select('not element H')
        tar_heavy = tar.select('not element H')
        print src_heavy.numAtoms()
        print tar_heavy.numAtoms()


        if src_heavy.numAtoms()!=tar_heavy.numAtoms():
            print 'Can\'t tell RMSD because number of Atoms are not same here!'
            return -1
        src_coord = src_heavy.getCoords()
        tar_coord = tar_heavy.getCoords()

        try:
            if benchmark is None:
                return np.sqrt(((src_coord - tar_coord) ** 2).mean())
            else:
                # align coordinates here
                src_coord = benchmark
                return np.sqrt(((src_coord - tar_coord) ** 2).mean())
        except:
            return 0

    def _calcRMSF(self,src,frames):
        return 0

    def _calcQ(self, src,tar,benchmark=None):
        '''

        Compute the fraction of native contacts according the definition from
        Best, Hummer and Eaton
        :param src:
        :param tar:
        :param benchmark:
        :return:
        '''
        src_heavy = src.select('not element H')
        tar_heavy = tar.select('not element H')
        receptor_heavy = self.receptor.select('not element H')

        if src_heavy.numAtoms()!=tar_heavy.numAtoms():
            print 'Can\'t tell RMSD because number of Atoms are not same here!'
            return -1
        src_coord = src_heavy.getCoords()
        tar_coord = tar_heavy.getCoords()
        receptor_coord = receptor_heavy.getCoords()

        if benchmark is None:
            src_coord= benchmark
        return native_contact(receptor_coord, src_coord, [tar_coord])[0]


    def bundle_autodock_file(self,ResId,score_only=False,src_ResId=None):

        #if self.heterodict[ResId]['file_generated']==True:
        #    return
        try:
            PDB= self.PDBname
            naming = '{}_{}'.format(PDB, ResId)
            middle= self.heterodict[ResId]['center']
            self.heterodict[ResId]['file_generated'] = True
            pdb_store_dir = self.store_dir
            #prepare files:
            if src_ResId is not None:
                filename2 = pdb_store_dir+'/{}/{}_{}_receptor.pdb'.format(src_ResId, PDB, ResId)
            else:
                filename2 = pdb_store_dir+'/{1}/{0}_{1}_receptor.pdb'.format(PDB, ResId)
            print filename2
            pd.writePDB(filename2, self.heterodict[ResId]['protein'])
            # pdb_to_mol2(filename2, ''.join(filename2.split('.')[:-2]) + '.mol')
            if src_ResId is not None:
                filename2 = pdb_store_dir+'/{}/{}_{}_complex.pdb'.format(src_ResId, PDB, ResId)
            else:
                filename2 = pdb_store_dir+'/{1}/{0}_{1}_complex.pdb'.format(PDB, ResId)
            if score_only==False:
                pd.saveAtoms(self.heterodict[ResId]['protein'], filename=os.path.join(pdb_store_dir,'temp.ag.npz'))
                atomgroup = pd.loadAtoms(os.path.join(pdb_store_dir,'temp.ag.npz'))
                pd.writePDB(filename2, self.heterodict[ResId]['ligand'] + atomgroup)
            #print filename2
            # Do autogrid mapgeneration:
            if src_ResId is None:
                ligand_filename = os.path.join(pdb_store_dir, ResId +'/' + naming + '_ligand.pdb')
                receptor_filename = os.path.join(pdb_store_dir, ResId +'/' + naming + '_receptor.pdb')
                complex_filename = os.path.join(pdb_store_dir, ResId +'/' + naming + '_complex.pdb')
            else:
                ligand_filename = self.heterodict[ResId]['filename']
                receptor_filename = os.path.join(pdb_store_dir, src_ResId + '/' + naming + '_receptor.pdb')
                complex_filename = os.path.join(pdb_store_dir, src_ResId + '/' + naming + '_complex.pdb')

            fake_ligand_filename = os.path.join(temp_pdb_PREFIX, 'fake-ligand.pdb')
            self.heterodict[ResId]['vina_score'] = do_auto_vina_score(os.path.join(pdb_store_dir, src_ResId),
                                                                      receptor_filename, ligand_filename, middle)


        except:
            self.heterodict[ResId]['vina_score'] = 'NA'
            return

        box_num = int(np.ceil(self.BOX_range / self.BOX_size))
        print box_num

        try:
            if score_only==False:
                print 'here'
                self.heterodict[ResId]['gridmap_protein'] = do_auto_grid(os.path.join(pdb_store_dir, src_ResId),receptor_filename, fake_ligand_filename,
                                                                     center=middle, BOX_size=self.BOX_size,BOX_num=box_num)
                self.heterodict[ResId]['gridmap_ligand'] = do_auto_grid(os.path.join(pdb_store_dir, src_ResId),ligand_filename, fake_ligand_filename,
                                                                    center=middle, BOX_size=self.BOX_size,BOX_num=box_num)
                self.heterodict[ResId]['gridmap_complex'] = do_auto_grid(os.path.join(pdb_store_dir, src_ResId),complex_filename, fake_ligand_filename,
                                                                     center=middle, BOX_size=self.BOX_size,BOX_num=box_num)
        except:
            return




    def find_similar_target(self,sdf_filedir,**kwargs):
        '''
        Find the ligands that is highly possible to be the same compounds
        the default confidence is 0.85
        the score was based on tanimoto scoring method
        :param sdf_filedir: where the source sdf file is. In theory, if we are using openbabel
                            it is ok even if the file is not sdf, but it should only contain single
                            molecules, other wise this function cannot get right result
        :param kwargs:
        :return:
        '''

        assert isinstance(sdf_filedir,str)
        if not os.path.exists(sdf_filedir) or sdf_filedir.split('.')[-1]!='sdf':
            raise IOError('Please use a right location, {} is not a legal file name of sdf file'.format(sdf_filedir))

        possible_ones=[]

        for k,v in self.heterodict.items():
            try:
                command = os.popen('babel -d {} {} -ofpt -xfFP4'.format(sdf_filedir,v['filename']))
                ls= command.read()
                #print ls
                cp = re.split('=|\n', ls)[2]
                print 'Similarity: {}'.format(cp)
            except:
                #raise TypeError
                with open('error.txt','a') as f:
                    f.write(self.PDBname+'\n')
                logging.warning('Babel encountered a problem at pdb {} ligand {}'.format(self.PDBname, v['filename']))
                cp = 0


            #print cp
            if float(cp) >= 0.85:
                possible_ones.append({'cp':cp,'id':k})

        return possible_ones


    def pick_one(self, ResId, **kwargs):
        return self.heterodict[ResId] or None

    def list_ResId(self):
        return self.heterodict.keys()

    def clean_temp_data(self):
        '''
        delete all files except '.pdb' '.mol', '.map' and name.pdbqt
        :return:
        '''
        '''exclude = ['pdb','mol','map']
        list_dirs = os.walk('data/'+self.PDBname)
        for root, dirs, files in list_dirs:
            for f in files:
                if f.split('.')[-1] not in exclude:
                        os.remove(os.path.join(root,f))
                if f.split('.')[-1] == 'map':
                    if f.split('.')[-2] not in electype:
                        os.remove(os.path.join(root, f))
        '''
        os.system('rm -r '+ os.path.join(temp_pdb_PREFIX, self.PDBname))

    def create_patch_file(self,ResId,dir='PDB'):
        '''
        Create the bundled result files in a nicer way.
        :param ResId:
        :param dir:
        :return:
        '''

        def copyFiles(src, tar):
            if not os.path.exists(tar) or \
                    (os.path.exists(tar) and (os.path.getsize(tar) != os.path.getsize(tar))):
                open(tar, "wb").write(open(src, "rb").read())

        real_loc = os.path.join(dir,self.PDBname)

        if not os.path.exists(real_loc):
            os.mkdir(real_loc)

        # Step 0: get original pdb file
        filename = self.PDBname+'.pdb'
        copyFiles('data/'+self.PDBname+'/'+filename,os.path.join(real_loc,filename))

        # Setp 1: first get receptor's file
        ligand_loc = os.path.join(real_loc,str(ResId))



        pass

    def bundle_result_dict(self,ResId,src_ResId=None):
        '''
        Render results into one_line string which contains docking score and some other infomation from pdb-ligand pair
        :param ResId:
        :return:p
        '''
        dict = self.heterodict[ResId]
        Remark_dict = collections.OrderedDict()
        self.bundle_autodock_file(ResId, score_only=True, src_ResId=src_ResId)

        Remark_dict['PDBname'] = self.PDBname
        Remark_dict['PDBResId'] = dict['id']
        Remark_dict['center'] = list_formatter(dict['center'])
        Remark_dict['rotation'] = list_formatter(dict['rotation'])
        Remark_dict['Resolution(A)'] = self.resolution
        Remark_dict['RMSF'] = dict['RMSF']
        Remark_dict['Contact Similarity'] = dict['Contact Similarity']

        try:
            Remark_dict['Autovina_Affinity(kcal/mol)'] = dict['vina_score']['Affinity']
            Remark_dict['Autovina_gauss1'] = dict['vina_score']['gauss 1']
            Remark_dict['Autovina_gauss2'] = dict['vina_score']['gauss 2']
            Remark_dict['Autovina_repulsion'] = dict['vina_score']['repulsion']
            Remark_dict['Autovina_hydrophobic'] = dict['vina_score']['hydrophobic']
            Remark_dict['Autovina_Hydrogen'] = dict['vina_score']['Hydrogen']
        except:
            return Remark_dict

        return Remark_dict


    def bundle_result(self,ResId,score_only=False,src_ResId=None):
        '''
        Render results into full vectors which contains info from pdbs
        With the order:
        PDBname PDBtype PDB_ligand_name PDB_ligand_resIndex center rotation resolution autodock_vina_para(*6) PDBsequence atom_vector
        :param ResId:
        :return:
        '''

        info_line=[]
        self.pdb_type = self.get_pdb_type()

        self.bundle_autodock_file(ResId,score_only,src_ResId=src_ResId)
        #self.create_patch_file(ResId,dir='PDB')
        naming = '%s_%s' %(self.PDBname,ResId)

        dict = self.heterodict[ResId]
        info_line.append(self.PDBname)
        info_line.append(self.pdb_type)
        info_line.append(dict['Resname'])
        info_line.append(dict['id'])
        info_line.append(list_formatter(dict['center']))
        info_line.append(list_formatter(dict['rotation']))
        info_line.append(dict['vina_score']['Affinity'])
        info_line.append(dict['vina_score']['gauss 1'])
        info_line.append(dict['vina_score']['gauss 2'])
        info_line.append(dict['vina_score']['repulsion'])
        info_line.append(dict['vina_score']['hydrophobic'])
        info_line.append(dict['vina_score']['Hydrogen'])
        info_line.append(self.resolution)
        info_line.append('1')
        info_line.append('0')
        info_line.append(self.sequence)
        info_line.append(list_formatter(dict['raw_vector']))

        if src_ResId is not None:
            store_dir = os.path.join(self.store_dir,src_ResId)
        else:
            store_dir = os.path.join(self.store_dir,ResId)

        if dict['gridmap_protein']!='NA':
            #for index in range(8):
            #    info_line.append(self.PDBname+'_'+dict['id']+'_receptor.'+electype[index]+'.map')
            info_line += map(list_formatter,fetch_gridmaps(store_dir,naming+'_receptor'))
        else:
            info_line+= ['NA']*8

        if dict['gridmap_ligand']!='NA':
            #for index in range(8):
            #    info_line.append(self.PDBname+'_'+dict['id']+'_ligand.'+electype[index]+'.map')
            print dict['filename'][:-3]
            info_line += map(list_formatter,fetch_gridmaps(store_dir,dict['filename'][:-4]))
        else:
            info_line+= ['NA']*8

        if dict['gridmap_complex']!='NA':
            #for index in range(8):
            #    info_line.append(self.PDBname+'_'+dict['id']+'_complex.'+electype[index]+'.map')
            info_line += map(list_formatter,fetch_gridmaps(store_dir,naming+'_complex'))
        else:
            info_line+= ['NA']*8

        return info_line


    def add_ligand(self,ligand_pdb_file,ResIndex, count_index,OUT=True,benchmark=None):
        '''
        Add ligands on to pdb. The result should be generated by docking, otherwise it will get some strange result.
        :param ligand_pdb_file:
        :return:
        '''
        try:
            parse = pd.parsePDB(ligand_pdb_file)

        except:
            #raise IOError
            logging.warning('cannot add ligang file on PDB {}'.format(self.PDBname))
            return
        self.bundle_ligand_data(parse,fake_ligand=True,OUT=OUT,compare_ResId_native=ResIndex,
                                Id_suffix=str(count_index),filename=ligand_pdb_file,benchmark=benchmark)



    def add_ligands(self,ligand_file,suffix=None,benchmark_file=None):
        '''
        Specialized only to generate data from Xiao's docking result and then convert them back
        :param ligand_file:
        :return:
        '''
        SYMBOL ='@<TRIPOS>MOLECULE'
        print ligand_file,suffix,benchmark_file
        try:
            if benchmark_file is not None:
                try:
                    parse = pd.parsePDB(benchmark_file).select('not element H')
                    bench_coord= parse.getCoords()
                except:
                    bench_coord= None

            fixfilename = ligand_file.split('/')[-1]
            residue_index = fixfilename.split('_')[1]

            self.get_residue_onfly(residue_index)

            pdbname = fixfilename.split('_')[0]
            count = 0
            with open(ligand_file,'rb') as f:
                for line in f.readlines():
                    if SYMBOL in line:
                        count+=1
            print count
            if suffix is not None:
                filename = "".join(fixfilename.split('.')[:-1])
                filename = filename + "_" + suffix + "_"
            else:
                filename = "".join(fixfilename.split('.')[:-1])
                filename = filename + "_"
            filedir = os.path.join(self.store_dir,residue_index+'/'+filename)
            print 'babel {} -opdb {}.pdb -m'.format(ligand_file,filedir)
            os.system('babel {} -opdb {}.pdb -m'.format(ligand_file,filedir))
            time.sleep(5)
            os.system('babel {} -omol2 {}.mol -m'.format(ligand_file, filedir))

            if suffix is None:
                result_filename= os.path.join(result_PREFIX,filename[:-1]+'.mol')
            else:
                result_filename = os.path.join(result_PREFIX, suffix+'/'+ filename[:-1] + '.mol')
            #if os.path.exists(result_filename):
            #    if os.path.getsize(result_filename) > 12000:
            #        print '%s_%s already done!' % (pdbname,residue_index)
            #        return

            with open(result_filename,'wb') as w:
                for i in range(count):
                    self.add_ligand(filedir+str(i+1)+'.pdb',ResIndex=residue_index,count_index=i+1,benchmark=bench_coord)
                    pdbdict = self.bundle_result_dict(residue_index+'_'+str(i+1),src_ResId= residue_index)
                    #print pdbdict
                    comment = 'Remark:'
                    if pdbdict is not None:
                        for k, v in pdbdict.items():
                            # print k,v
                            # print comment
                            comment = comment + '_{' + k + ':' + str(v)+ '}'
                    comment +='}'
                    w.write('# '+comment+'\n')
                    #print comment
                    with open(filedir+str(i+1)+'.mol','rb') as f:
                        tag= False
                        content =''
                        for line in f.readlines():

                            if tag:
                                tag= False
                                continue
                            if SYMBOL in line:
                                tag= True
                            content+=line
                        w.write(content)


        except:
            return False

    def __repr__(self):
        print self.PDBname+'({} hetero parts found)'.format(len(self.heterodict.keys()))

    def __del__(self):
        files = self.store_dir
        if os.path.exists(files):
            os.system('rm -r ' + files)

