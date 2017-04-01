import os
import sys
import config

mkdir = lambda path: os.system('mkdir -p {}'.format(path))

def log(log_file, log_content, head=None, lock=None):
    """
    write down log information
    :param log_file: name of log file
    :param log_content: string or list of string, log content
    :param head: head for csv or tsv output file
    :param lock: multiprocess lock
    :return:
    """

    lock.acquire()

    if isinstance(log_content, str):
        log_content = [log_content]

    mkdir(config.log_folder)

    log_file_path = os.path.join(config.log_folder, log_file)
    if not os.path.exists(log_file_path) and not head is None:
        with open(log_file_path, 'w') as fout:
            fout.write(head+'\n')
    with open(log_file_path, 'a') as fout:
        for cont in log_content:
            if type(cont).__name__ == 'list':
                print cont
            fout.write(cont+'\n')

    lock.release()

class smina_param:

    kw_options = [
        'receptor',
        'ligand',
        'out',
        'flex',
        'flexres',
        'flexdist_ligand',
        'flexdist',
        'center_x',
        'center_y',
        'center_z',
        'size_x',
        'size_y',
        'size_z',
        'autobox_ligand',
        'autobox_add',
        'scoring',
        'custom_scoring',
        'minimize_iters',
        'approximation',
        'factor',
        'force_cap',
        'user_grid',
        'user_grid_lambda',
        'out_flex',
        'log',
        'atom_terms',
        'cpu',
        'seed',
        'exhaustiveness',
        'num_modes',
        'energy_range',
        'min_rmsd_filter',
        'addH',
        'config'
    ]

    arg_options = [
        'no_lig',
        'score_only',
        'local_only',
        'minimize',
        'randomize_only',
        'accurate_line',
        'print_terms',
        'print_atom_types',
        'atom_term_data',
        'quiet',
        'flex_hydrogen',
        'help',
        'help_hidden',
        'version'
    ]

    def __init__(self):
        pass
        
    def set_smina(self, smina):
        self.smina = smina

    def set_name(self, name):
        self.name = name

    def load_param(self, *arg, **kw):
        self.args = arg
        self.kwargs = kw

    def make_command(self, *arg, **kw):
        cmd = self.smina
        for a in self.arg_options:
            if arg and a in arg:
                cmd += ' --'
                cmd += a
            elif a in self.args:
                cmd += ' --'
                cmd += a

        for key in self.kw_options:
            if kw and key in kw.keys():
                cmd += ' --'
                cmd += key
                cmd += ' '
                cmd += str(kw[key])
            elif key in self.kwargs.keys():
                cmd += ' --'
                cmd += key
                cmd += ' '
                cmd += str(self.kwargs[key])

        return cmd
    
    