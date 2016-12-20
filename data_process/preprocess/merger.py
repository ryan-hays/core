from Config import result_PREFIX,merged_Filename
import os,io
import csv

def merge_the_result(filenames):
    totalfile = file(os.path.join(result_PREFIX,merged_Filename),'wb')
    out = csv.writer(totalfile)

    # the flag to determine whether the first line is output or not
    Never_OUTPUT= True

    count = 0
    lines =0

    f = open('report.txt', 'wb')

    # write legal files
    for filedir in sorted(filenames):
        print filedir
        if filedir== merged_Filename or filedir.split('.')[-1]!='csv':
            continue

        # csv writer
        csvfile= file(os.path.join(result_PREFIX,filedir),'rb')
        o = csv.reader(csvfile)

        for line in o:
            #First line
            if line[0]=='NAME':
                if Never_OUTPUT:
                    out.writerow(line)
                    Never_OUTPUT= False
                continue

            #Not the first line
            out.writerow(line)
            lines+=1

        count+=1
        print 'Successfully merge {}'.format(filedir)
        f.write('{} lines are valid in {}\n'.format(lines-1,filedir))

        csvfile.close()

    print '{} files and {} lines are merged into one called {}'.format(count,lines,merged_Filename)

    f.write('{} files and {} lines are merged into one called {}'.format(count,lines,merged_Filename))
    f.close()
    totalfile.close()

if __name__=='__main__':

    filenames= os.listdir(result_PREFIX)

    result_filename = []
    for filename in filenames:
        if 'filter' in filename:
            result_filename.append(filename)
    merge_the_result(result_filename)
