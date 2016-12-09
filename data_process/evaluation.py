import numpy as np
import os,sys,re,random,copy

'''
evaluation the submission use map

'''



def parse(input_file):
    '''
    parse input into a dict, it's convenient to use dict calculate map
    :param input_file: path of file
    :return: a dict key is receptor's code, value is ligands' code
    '''

    input_dict = {}
    with open(input_file) as fr:
        fr.readline()
        for line in fr:
            key,value = line.strip('\n').split(',')
            value = value.split(' ')
            input_dict[key] =value

    return input_dict

def mean_average_pricision(solution,submission,n=None):
    '''
    Calculate mean average precision
    :param solution: dict
                        key: receptor's id
                        value: list of ranked ligands' id
    :param submission: dict
                        key: receptor's id
                        value: list of ranked ligands' id
    :param n:
    :return:
    '''
    aps = []
    for key in solution.keys():
        ap = 0
        if not key in submission.keys():
            # if failed to predict, return a random ranked value
            answer = solution[key]
            prediction = copy.copy(answer)
            random.shuffle(prediction)

            if len(answer)<n:
                ap = 1
                aps.append(ap)
            else:
                for i in range(n):
                    ligand = prediction[i]
                    location = answer.index(ligand) if (ligand in answer) else -1
                    ap += (i+1.0)/(location + 1.0) if location >= 0 else 0


                ap/= n
                aps.append(ap)

        elif len(submission[key])==0:
            ap = 0
            aps.append(ap)
        else:

            answer = solution[key]
            prediction = submission[key]
            if len(answer)<n:
                ap = 1
                aps.append(ap)

            else:
                for i in range(n):
                    ligand =prediction[i]
                    location = answer.index(ligand) if (ligand in answer) else -1
                    ap += (i+1.0)/(location + 1.0) if location >= 0 else 0

                ap /=n
                aps.append(ap)



    return sum(aps)/len(aps)





def eval():
    solution = './solution.txt'
    submission = './logs-155999_submission.txt'
    print mean_average_pricision(parse(solution),parse(submission))

