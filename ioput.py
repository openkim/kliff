 
class InputKeywords:
    '''
    Class to deal with controlling keywords. 
    '''
    def __init__(self):
        '''
        List all the supported keywords here, including name, value and type.
        '''
        self.keywords = {'name':[], 'type':[], 'value':[]} 
        self.add_keyword('stepsize',        'float')
        self.add_keyword('maxsteps',        'int', 10000)
        self.add_keyword('include',         'str')
        self.add_keyword('globalopt',       'boolean')
        


    def add_keyword(self, name, data_type, value=None):
        self.keywords['name'].append(name)
        self.keywords['type'].append(data_type)
        self.keywords['value'].append(value)

    
    def remove_comments(self, lines):
        'Remove lines starting with # and content after #.'
        processed_lines = []
        for line in lines:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            if '#' in line:
                line = line[0:line.index('#')] 
            processed_lines.append(line)
        return processed_lines


    def read(self, fname='./kimfit.in'):
        'Read keywords from file.'
        with open(fname, 'r') as fin:
            infile = fin.readlines()
        infile_hash_removed= self.remove_comments(infile)
        for line in infile_hash_removed:
            line = line.split()
            name = line[0].lower(); value = line[1]
            if name in self.keywords['name']:
                idx = self.keywords['name'].index(name)
                if self.keywords['type'][idx] == 'float':
                    self.keywords['value'][idx] = float(value)
                elif self.keywords['type'][idx] == 'int':
                    self.keywords['value'][idx] = int(value)
                elif self.keywords['type'][idx] == 'boolean':
                    if value.lower() == 'ture' or value.lower() == 't':
                        self.keywords['value'][idx] = True 
                    else:
                        self.keywords['value'][idx] = False
                elif self.keywords['type'][idx] == 'str':
                    self.keywords['value'][idx] = value


    def write(self, fname='./kimfit.log'):
        'Write keywords to file.'
        with open(fname, 'w') as fout:
            for name, value in zip(self.keywords['name'], self.keywords['value']):
                if value != None:
                    fout.write('{}  {}\n'.format(name, value)) 







class TrainingSet:
    '''
    Class to read and store training set. 
    '''
    
    def __init__(self):
        self.numconfig = 1

    def read(self, fname='./train.xyz'):
        pass
        


# test
if __name__ == '__main__':
    keys = InputKeywords()
    keys.read('./kimfit.in')
    keys.write()




