import csv

def parser():
    csvr = csv.reader(open('fer2013.csv'))
    header = next(csvr)
    rows = [row for row in csvr]
    
if __name__ == "__main__":
    data = parser()
    print data[0]
    print data[1]
    trn = [row[:-1] for row in rows if row[-1] == 'Training']
#     print trn[0]
#     csv.writer(open('test.csv', 'w+')).writerows([header[:-1]] + trn)
    print(len(trn))