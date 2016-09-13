import argparse

class InputKeywords:
    '''
    Class to deal with controlling keywords. 
    '''
    def __init__(self):
        '''
        List all the supported keywords here, including internal name, 
        data type, default value.   
        '''
        self.keywords = dict()
        
        # names
        self.add_keyword('modelname',       'str')
        self.add_keyword('trainingset',     'str')
        self.add_keyword('testset',         'str')

        # optimization related
        self.add_keyword('maxsteps',        'int',          10000)
        self.add_keyword('regularity',      'boolean',      False)
        self.add_keyword('global',          'boolean',      False)
        self.add_keyword('optimize',        'boolean',      True)

        # Levenberg-Marquardt
        self.add_keyword('lm.lambda',       'float',        1.12)


    def add_keyword(self, kwd, dtype, value=None, readin=False, name=None):
        self.keywords[kwd] = {'type':dtype, 'value':value, 'readin':readin, 'name':name}
        
    def read(self):
        '''Read keywords from file.'''
        fname, = parse_arg()
        all_keywords = self.keywords.keys()
        with open(fname, 'r') as fin:
            infile = fin.readlines()
        infile_hash_removed = remove_comments(infile)
        for line in infile_hash_removed:
            line = line.split()
            kwd = line[0]
            if len(line) >= 2:     
                value = line[1]
            else:
                raise InputError('value missing for keyword "{}" in file '
                                 '"{}".'.format(kwd, fname))
            kwd_lower = kwd.lower()
            if kwd_lower not in all_keywords:
                raise InputError('unrecognized keyword "{}" in file "{}".'.format(kwd, fname))
            else:
                expected_type = self.keywords[kwd_lower]['type']  
                if expected_type == 'float':
                    try: 
                        self.keywords[kwd_lower]['value'] = float(value)
                    except ValueError as err:
                        raise  ValueError('{}.\ndata type of "{}" in keyword file should be '
                                          '"{}".'.format(err, kwd, expected_type))
                elif expected_type == 'int':
                    try: 
                        self.keywords[kwd_lower]['value'] = int(value)
                    except ValueError as err:
                        raise  ValueError('{}.\ndata type of "{}" in keyword file should be '
                                          '"{}".'.format(err, kwd, expected_type))
                elif expected_type == 'boolean':
                    if value.lower() == 'true' or value.lower() == 't':
                        self.keywords[kwd_lower]['value'] = True
                    else:
                        self.keywords[kwd_lower]['value'] = False
                elif expected_type == 'str':
                    self.keywords[kwd_lower]['value'] = value
                # this keyword has value read in from file, record it
                self.keywords[kwd_lower]['readin'] = True 
                self.keywords[kwd_lower]['name'] = kwd


    def echo_readin(self):
        '''Echo the keywords that are read in from file.'''
        print '='*80
        print 'Keywords read from input file.\n'
        for name in self.keywords: 
            if self.keywords[name]['readin']:
                print '{:15} {}'.format(self.keywords[name]['name'],
                                        self.keywords[name]['value'])

    def get_value(self, key):
        ''' Get the "value" of keywords "key". '''
        return self.keywords[key]['value']

   


def parse_arg():
    '''Parse stdin argument.'''
    parser = argparse.ArgumentParser(description='OpenKIM potential model fitting program.')
    parser.add_argument('file_in', type=str, help='Name of input file that contains the '
                                                  'keywords of the program.')
    args = parser.parse_args()
    return [args.file_in] 

 
def remove_comments(lines):
    '''Remove lines in a string list that start with # and content after #.'''
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        if '#' in line:
            line = line[0:line.index('#')] 
        processed_lines.append(line)
    return processed_lines

          
class InputError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value 




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
    keys.read()
    keys.echo_readin()
    keys.write('./kimfit.log')



