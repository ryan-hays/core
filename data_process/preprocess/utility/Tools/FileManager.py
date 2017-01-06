'''
This is a simple class method to simplify access of file property

Since the program will have batch file IO tasks, it is useful to

get access of the file info. in a nice, decent way.



'''

import os.path as ospath
from shutil import copyfile

class FileManager:
    '''
    It facilitates the process to get file attributes

    It receives the input and manage the file direction, so this will heavily reduce the annoy scripts to
    locate the file direction (while doing data cleaning)

    '''


    def __init__(self,filedir,filename):
        '''
        Assuming the file location was given by filename+ directory
        :param filename: xxxx.pdb
        :param filedir:  /usr/system/data/
        :return:
        '''
        self._file_abspath = ospath.join(filedir, filename)
        self._file_dir, self._file_name = ospath.split(self._file_abspath)
        self._file_text, self._file_ext = ospath.splitext(self._file_name)

        # This is a ridiculous inner function error
        if self._file_ext[0]=='.':
            self._file_ext = self._file_ext[1:]


    @property
    def name(self):
        return self._file_name

    @property
    def dir(self):
        return self._file_dir

    @property
    def abspath(self):
        return self._file_abspath

    @property
    def text(self):
        return self._file_text

    @property
    def ext(self):
        return self._file_ext



    def exists(self):
        '''
        check if the file exists or the path is valid
        :return:
        '''
        try:
            return ospath.exists(self._file_abspath)
        except:
            return False

    def copy(self,tar_abspath, wrap_manager=False, overwrite= False):
        '''

        :param tar_abspath:  The absolute path for the file
        :param wrap_manager:  Whether the program will return a new FileManager class for this copy file,
        sometimes we first create the FileManager class and then get the copy.
        :param overwrite: if set False, the program will check if the path is valid nad
        :return:
        '''

        if not self.exists():
            raise IOError("This FileManager point to an illegal path or the file has been deleted");
        path,filename= ospath.split(tar_abspath)
        if not ospath.exists(path):
            raise IOError("The directory does not exist, please create the directory first!");
        if ospath.exists(tar_abspath):
            if overwrite:
                print("You choose to overwrite the file, this is irreversible!")
                copyfile(self._file_abspath,tar_abspath)
            else:
                print("File will not overwrite! Set overwrite=True in function to overwrite the existing file!")
        if wrap_manager==True:
            return FileManager(tar_abspath);
        else:
            return True;

    def destroy_file(self):
        '''
        Delete the file the class point to (if applicable)
        :return:
        '''
        if self.exists():
            from os import remove
            remove(self._file_abspath);
        else:
            print("File not exists, this is an invalid function call");



if __name__=="__main__":
    A=FileManager('/media/wy/data/','a.pdb')
    print(A.exists())
    A.destroy_file()