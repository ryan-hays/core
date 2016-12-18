
import platform
import os
from numpy.random import choice

# change these if you want the output to be somewhere else
cluster_output_file_location = '.'

def do_one_clustering(input_fasta_file):
    '''
    We appreciate the contribution from MMseq2, which offers people an ultra fast sequence database scheme
    for more information , please visit their github page.

    https://github.com/soedinglab/mmseqs2

    In other words, this script will fail on Windows.

    :param input_fasta_file: where the program get the fasta result, it can be either aligned data
     or nonaligned sequence stirng
    :return: use mmseq2 to finish a dirty and fast sequence clustering

    '''
    import subprocess
    import shlex
    if 'Linux' not in platform.system():
        print 'This might not be an linux machine. Mac might be ok but please do not use Windows for this scirpt.'

    #quick and dirty way to do sequence clustering
    #This will be changed in future
    subprocess.call(shlex.split('./mmseqs.sh %s %s'%(input_fasta_file,
                                                         os.path.join(cluster_output_file_location,'clustered.tsv'))))


    return True

def parse_group_result(cluster_file,result_file="result_group.txt"):
    '''
    Parse results from mmseq2 and write it into a file with designate format
    The format will be:
        First line: a postitive interger N with the number of groups
        Next N lines: each line has several PDB ids which means they are in same group

    :param cluster_file: file location for tsv cluster file
    :param result_file: file location for writing result
    :return: write results into result_file and return the list of result (group result)
    '''
    representative = "XXXX"
    list= []
    count=0

    with open(cluster_file,'r') as f:
        for each_line in f.readlines():
            first= each_line.split('\t')[0]
            second = each_line.split('\t')[1].rstrip('\n')
            print first,second
            if representative != first:
                count+=1
                list.append([])
                representative = first
            list[-1].append(second)

    with open(result_file,'w') as w:
        w.write(str(count)+'\n')
        for each_group in list:
            for each in each_group[:-1]:
                w.write(each+' ')
            w.write(each_group[-1]+'\n')
    return list


def create_group_from_FASTA(fasta_file,result_file='result_group.txt'):
    '''

    :param fasta_file:
    :return:
    '''

    tag = do_one_clustering(fasta_file)
    if tag==False:
        print "Input Error! Try again"
        return []

    Ans = parse_group_result(os.path.join(cluster_output_file_location,'clustered.tsv'),result_file)

    return Ans

def select_subgroup_from_variable(group, max_element_number=0):
    '''
    Select some results and make sure each group will have at most one element to be selected.
    Note this is the uniform sampling for each group.

    :param max_element_number: how many elements will pick up at most. by default 0 means
    the maximum number, i.e. the total group number
    :return: a list file in which PDB IDs are provided
    '''
    length= len(group)
    subgroup = []
    if max_element_number==0:
        max_element_number=length
    else:
        if max_element_number>length:
            print 'This is out of range, there will be only ' \
                  '%s groups and all these will be returned' % str(length)
            max_element_number= length

    if max_element_number==length:
        for each_group in group:
            subgroup.append(choice(each_group))
    else:
        for each_group in choice(group,max_element_number):
            subgroup.append(choice(each_group))

    return subgroup

def select_subgroup_from_file(filename, max_element_number=0):
    '''

    :param filename:
    :param max_element_number:
    :return:
    '''
    list = []
    first_line= False
    try:
        with open(filename, 'r') as f:
            for each_line in f.readlines():
                if first_line == False:
                    first_line = True
                    continue
                group = each_line.split(' ')
                group[-1]=group[-1].rstrip('\n')
                list.append(group)
        return select_subgroup_from_variable(list,max_element_number=max_element_number)
    except:
        raise IOError('cannot parse such file!')



if __name__ == '__main__':
    '''
    To Use this script:
    1. run create_group_from_FASTA once and get the variable(optional)
    2. call as many times of either select_subgroup_from_variable or select_subgroup_from_file as
    you want
    '''
    Group = create_group_from_FASTA('full.fasta')
    #Group = parse_group_result(os.path.join(cluster_output_file_location,'clustered.tsv'))
    A= select_subgroup_from_variable(Group)
    B= select_subgroup_from_file('result_group.txt',max_element_number=100)
    print A
    print B
