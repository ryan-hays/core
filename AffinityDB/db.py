import os
import sys
import sqlite3
import config
import time
from config import lock
from utils import lockit

class database:

    def __init__(self):
        self.db_path = config.db_path
        self.connect_db()

    def connect_db(self):
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
        self.create_tabel()


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

        #print stmt

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
        
    def create_tabel(self):
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

        # overlap
        stmt.append( '''
        CREATE TABLE OVERLAP
        (DOCKED_LIGAND      TEXT NOT NULL,
        CRYSTAL_LIGAND      TEXT NOT NULL,
        POSITION            INTEGER NOT NULL,
        OVERLAP_RATIO       READ NOT NULL,
        PRIMARY KEY(DOCKED_LIGAND, CRYSTAL_LIGAND, POSITION)
        );''')

        # overlap state
        stmt.append( '''
        CREATE TABLE OVERLAP_STATE
        (DOCKED_LIGAND      TEXT NOT NULL,
        STATE               INTEGER NOT NULL,
        PRIMARY KEY(DOCKED_LIGAND)
        );''')

        # rmsd 
        stmt.append( '''
        CREATE TABLE RMSD
        (DOCKED_LIGAND      TEXT NOT NULL,
        CRYSTAL_LIGAND      TEXT NOT NULL,
        POSITION             INTEGER NOT NULL,
        RMSD                READ NOT NULL,
        PRIMARY KEY(DOCKED_LIGAND, CRYSTAL_LIGAND, POSITION)
        );''')

        stmt.append( '''
        CREATE TABLE RMSD_STATE
        (DOCKED_LIGAND      TEXT NOT NULL,
        CRYSTAL_LIGAND      TEXT NOT NULL,
        STATE               INTEGER NOT NULL,
        PRIMARY KEY(DOCKED_LIGAND, CRYSTAL_LIGAND)
        );''')

        # native contace 
        stmt.append( '''
        CREATE TABLE NATIVE_CONTACE
        (DOCKED_LIGAND      TEXT NOT NULL,
        POSITION            INTEGER  NOT NULL,
        RATIO_4_0           REAL,
        RATIO_4_5           REAL,
        RATIO_5_0           REAL,
        RATIO_5_5           REAL,
        RATIO_6_0           REAL,
        PRIMARY KEY(DOCKED_LIGAND,POSITION)
        );''')

        for s in stmt:
            self.conn.execute(s)

        print "create all %d tabels" % len(stmt)