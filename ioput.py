 
class InputKeywords:
    '''
    Class to deal with controlling keywords. 
    '''
    def __init__(self):
        '''
        List all the supported keywords here, including name, default value and type.
        '''
        self.keywords = dict()
        
        # file names
        self.add_keyword('file_model',      'str',          './KIMmodel.txt')
        self.add_keyword('file_training',   'str',          './train.xyz')
        self.add_keyword('file_test',       'str')

        # optimization related
        self.add_keyword('maxsteps',        'int',          10000)
        self.add_keyword('regularity',      'boolean',      False)
        self.add_keyword('global',          'boolean',      False)
        self.add_keyword('optimize',        'boolean',      True)

        # Levenberg-Marquardt
        self.add_keyword('lm.lambda',       'float',        1.12)


    def add_keyword(self, name, dtype, value=None, readin=False):
        self.keywords[name] = {'type':dtype, 'value':value, 'readin':readin}
        
    
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


    def read(self, fname='kimfit.in'):
        'Read keywords from file.'
        keyword_names = self.keywords.keys()
        with open(fname, 'r') as fin:
            infile = fin.readlines()
        infile_hash_removed= self.remove_comments(infile)
        for line in infile_hash_removed:
            line = line.split()
            name = line[0]; value = line[1]
            lower_name = name.lower()
            if lower_name not in keyword_names:
                raise Exception('unrecognized keyword "{}" in file "{}".'.format(name, fname))
            else:
                expected_type = self.keywords[lower_name]['type']  
                if expected_type == 'float':
                    try: 
                        self.keywords[lower_name]['value'] = float(value)
                    except ValueError:
                        raise  ValueError('data type of "{}" in keyword file should be '
                                          '"{}".'.format(name, expected_type))
                elif expected_type == 'int':
                    try: 
                        self.keywords[lower_name]['value'] = int(value)
                    except ValueError:
                        raise  ValueError('data type of "{}" in keyword file should be '
                                          '"{}".'.format(name, expected_type))
                elif expected_type == 'boolean':
                    if value.lower() == 'true' or value.lower() == 't':
                        self.keywords[lower_name]['value'] = True
                    else:
                        self.keywords[lower_name]['value'] = False
                elif expected_type == 'str':
                    self.keywords[lower_name]['value'] = value
                # this keyword has value read in from file
                self.keywords[lower_name]['readin'] = True 


    def echo_readin(self):
        'Echo the keywords that are read in from file.'
        print '='*80
        print 'Keywords read from input file.\n'
        for name in self.keywords: 
            if self.keywords[name]['readin']:
                print '{}   {}'.format(name, self.keywords[name]['value'])



# test
if __name__ == '__main__':
    #keys = InputKeywords()
    #keys.read('develop_test/kimfit.in')
    #keys.echo_readin()

    class write_all_keywords(InputKeywords):
        def write(self, fname='./kimfit.log'):
            'Write all keywords to file.'
            with open(fname, 'w') as fout:
                for name in self.keywords: 
                    fout.write('{}   {}\n'.format(name, self.keywords[name]['value']))

    keys = write_all_keywords()
    keys.read('develop_test/kimfit.in')
    keys.echo_readin()
    keys.write('./kimfit.log')



