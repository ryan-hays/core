import os
import sys
import sqlite3
import config
import time
from config import lock
from utils import lockit
import csv
from collections import namedtuple, OrderedDict

table = namedtuple('Table',['name','columns','primary_key'])

tables = {
    'atom_num':table(*['atom_num',
        OrderedDict([('name','text'),('type','text'),('atom','integer')]),
        ['name']]),

    'rotable_bond':table(*['rotable_bond',
        OrderedDict([('ligand','text'),('rotable_bond','ingeter')]),
        ['ligand']]),

    'resolution':table(*['resolution',
        OrderedDict([('pdb','text'),('experiment','text'),('resolution','real')]),
        ['pdb']]),

    'split_state':table(*['split_state',
        OrderedDict([('pdb','text'),('state','integer'),('comment','text')]),
        ['pdb']]),

    'similarity':table(*['similarity',
        OrderedDict([('ligand_a','text'),('ligand_b','text'),('finger_print','text'),
            ('similarity','real')]),
        ['ligand_a,ligand_b']]),

    'overlap':table(*['overlap',
        OrderedDict([('docked_ligand','text'),('crystal_ligand','text'),
            ('position','integer'),('overlap_ratio','real')]),
        ['docked_ligand', 'crystal_ligand', 'position']]),

    'overlap_state':table(*['overlap_state',
        OrderedDict([('docked_ligand','text'),('crystal_ligand','text'),
            ('state','integer'),('comment','text')]),
        ['docked_ligand','crystal_ligand']]),

    'rmsd':table(*['rmsd',
        OrderedDict([('docked_ligand','text'),('crystal_ligand','text'),
            ('position','ingeter'),('rmsd','real')]),
            ['docked_ligand','crystal_ligand','position']]),

    'rmsd_state':table(*['rmsd_state',
        OrderedDict([('docked_ligand','text'),('crystal_ligand','text'),
            ('state','integer'),('comment','text')]),
        ['docked_ligand, crystal_ligand']]),

    'native_contact':table(*['native_contact',
        OrderedDict([('docked_ligand','text'),('position','integer'),
            ('ratio_4_0','real'),('ratio_4_5','real')]),
            ['docked_ligand','position']]),

    'native_contact_state':table(*['native_contact_state',
        OrderedDict([('docked_ligand','text'),('state','integer'),
            ('comment','text')]),
            ['docked_ligand']]),

    'dock_state':table(*['dock_state',
        OrderedDict([('docked_ligand','text'),('state','integer'),
            ('comment','text')]),
            ['docked_ligand']])
}

class database:

    def __init__(self):
        self.db_path = config.db_path
        self.tables = tables
        self.add_scoring_term_tabel()
        self.connect_db()

    def connect_db(self):
        print "connect to %s" % self.db_path
        self.conn = sqlite3.connect(self.db_path)
        self.connect = True

    def backup_db(self):
        
        backup_db_path = self.db_path.replace('.', '_'+time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime()) +'.')

        if os.path.exists(self.db_path):
            cmd = 'cp %s %s' % (self.db_path , backup_db_path)
            os.system(cmd)
            print "backup database %s" % backup_db_path

    def backup_and_reset_db(self):

        if os.path.exists(self.db_path):
            self.backup_db()
            
            cmd = 'rm %s' % self.db_path
            os.system(cmd)

        self.connect_db()
        self.create_table()


    @lockit
    def insert(self, tabel, values, head=None):
        
        sql_values = [ '(' + ','.join(value) + ')' for value in values ]

        stmt = 'INSERT INTO ' + tabel + ' '
        if not head is None:
            stmt += (','.join(head))
        stmt += ' VALUES '
        stmt += ','.join(sql_values)
        stmt += ';'

        if not self.connect:
            self.connect_db()
        try:
            self.conn.execute(stmt)
        except sqlite3.IntegrityError as e:
            print "Integrity Error ",
            print e
        except Exception as e:
            print e

        self.conn.commit()

    @lockit
    def insert_or_replace(self, tabel, values, head=None):
        
        db_value = lambda x:'"%s"' % x if type(x).__name__ == 'str' else str(x)

        db_values = [ map(db_value, value) for value in values ]
        print db_values
        
        sql_values = [ '(' + ','.join(value) + ')' for value in db_values ]
        
        print sql_values
        stmt = 'REPLACE INTO ' + tabel + ' '
        if not head is None:
            stmt += '('+ ','.join(head) + ')'
        stmt += ' VALUES '
        stmt += ','.join(sql_values)
        stmt += ';'

        print stmt

        if not self.connect:
            self.connect_db()
        try:
            self.conn.execute(stmt)
        except Exception as e:
            print e
        
        self.conn.commit()

    @lockit
    def insert_or_ignore(self, tabel, values, head):
        
        sql_values = [ '(' + ','.join(value) + ')' for value in values ]

        stmt = 'INSERT OR IGNORE INTO ' + tabel + ' '
        stmt += (','.join(head))
        stmt += ' VALUES '
        stmt += ','.join(sql_values)
        stmt += ';'

        if not self.connect:
            self.connect_db()
        try:
            self.conn.execute(stmt)
        except Exception as e:
            print e
        
        self.conn.commit()

    def create_table(self):
        
        for tab in tables.values():
            stmt = 'create table '+ tab.name + ' ('
            for key in tab.columns.keys():
                stmt += key + ' ' + tab.columns[key]
                if key in tab.primary_key:
                    stmt += ' not null ,'
                else:
                    stmt += ' ,'
            stmt += 'primary key(' + ','.join(tab.primary_key) + '));'
            
            self.conn.execute(stmt)

        print "Create all %d tables" % len(tables)        
            
        
    def create_tabel_old(self):
        """
        create the tabels that we gonna use
        """

        stmt = []
        # atom num tabel
        stmt.append( '''
        CREATE TABLE ATOM_NUM
        (NAME       TEXT NOT NULL,
        TYPE        TEXT NOT NULL,
        ATOM_NUM    INTEGER,
        PRIMARY KEY(NAME)
        );''')

        # rotable bond
        stmt.append( '''
        CREATE TABLE ROTABLE_BOND
        (LIGAND         TEXT NOT NULL,
        ROTABLE_BOND    INTEGER,
        PRIMARY KEY(LIGAND)
        );''')

        stmt.append('''
        CREATE TABLE RESOLUTION
        (PDB        TEXT NOT NULL,
        RESOLUTION  REAL NOT NULL,
        PRIMARY KEY(PDB)
        );''')

        # parse stat
        stmt.append( '''
        CREATE TABLE SPLIT_STATE
        (PDB        TEXT NOT NULL,
        STATE       INTEGER,
        COMMENT     TEXT,
        PRIMARY KEY(PDB)
        );''')


        # tanimoto similarity
        stmt.append( '''
        CREATE TABLE SIMILARITY
        (LIGAND_A       TEXT NOT NULL,
        LIGAND_B        TEXT NOT NULL,
        FINGER_PRINT    TEXT NOT NULL,
        SIMILARITY      READ NOT NULL,
        PRIMARY KEY(LIGAND_A, LIGAND_B)
        );''')

        stmt.append('''
        create table similarity
        (ligand_a       text not null,
        ligand_b        text not null,
        );''')

        # overlap

        stmt.append('''
        create table overlap
        (docked_ligand      text not null,
        crystal_ligand      text not null,
        position            integer not null,
        overlap_ratio       real,
        primary key(docked_ligand, crystal_ligand, position)
        );''')

        # overlap state

        stmt.append('''
        create table overlap_state
        (docked_ligand    text not null,
        state             integer,
        primary key(docked_ligand);
        );''')

        # rmsd 
        stmt.append('''
        create table rmsd
        (docked_ligand      text not null,
        crystal_ligand      text not null,
        position            integer,
        rmsd                real,
        primary key(docked_ligand, crystal_ligand, position)
        );''')

        stmt.append('''
        create table rmsd_state
        (docked_ligand      text not null,
        crystal_ligand      text not null,
        state               integer,
        comment             text,
        primary key(docked_ligand, crystal_ligand)
        );''')

        # native contace 
        stmt.append('''
        create table native_contact
        (docked_ligand      text not null,
        position            integer not null,
        ratio_4_0           read,
        ratio_4_5           real,
        ratio_5_0           real,
        ratio_5_5           real,
        ratio_6_0           real,
        primary key(docked_ligand, position)
        );''')

        stmt.append('''
        create table dock_state
        (docked_ligand      text not null,
        state               integer,
        comment             text,
        primary key(docked_ligand)
        );''')



        for s in stmt:
            self.conn.execute(s)

        print "create all %d tabels" % len(stmt)

    
    def add_scoring_term_tabel(self):
        """
        columns of this tabel will determined by 
        the content of config.scoring_tems 
        """

        
        columns = []
        columns.append(('ligand','text'))
        columns.append(('position','integer'))
        for row in open(config.scoring_terms):
            row = row.strip().split('  ')
            col = row[1].replace(',',' ')
            col = '"%s"' % col
            columns.append((col,'real'))

        self.tables['scoring_terms'] = table(*['scoring_terms',
            OrderedDict(columns),
            ['ligand','position']])

        
        
    

