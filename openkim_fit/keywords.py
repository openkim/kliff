##NOTE the interface has changed, and we do not need this file to parse input file
import argparse
from collections import OrderedDict
from error import InputError
from utils import remove_comments

class InputKeywords:
  '''
  Class to deal with controlling keywords.
  '''
  def __init__(self):
    '''
    List all the supported keywords here, including internal name,
    data type, default value.
    '''
    self.keywords = OrderedDict()
    self.blocks = dict()

    self.add_keyword('trainingset',     'str')
    self.add_keyword('testset',       'str')
    self.add_keyword('modelname',     'str')
    self.add_keyword('echo_avail_params', 'bool',   False)

    # optimization related
    self.add_keyword('maxsteps',      'int',    10000)
    self.add_keyword('regularity',      'bool',   False)
    self.add_keyword('global',        'bool',   False)
    self.add_keyword('optimize',      'bool',   True)

    # Levenberg-Marquardt
    self.add_keyword('lm.lambda',   'float',    1.12)

    # block values that will be delt with elsewhere
    self.add_block_keyword('modelparameters')


  def add_keyword(self, kwd, dtype, value=None, readin=False, name=None):
    self.keywords[kwd] = {'type':dtype, 'value':value, 'readin':readin, 'name':name}

  def add_block_keyword(self, kwd):
    self.blocks[kwd] = None

  def read(self):
    '''Read keywords from file.'''
    fname, = parse_arg()
    all_keywords = self.keywords.keys()
    with open(fname, 'r') as fin:
      lines = fin.readlines()
    lines_hash_removed = remove_comments(lines)
    lines_block_extracted = self.extract_block(lines_hash_removed)

    for line in lines_block_extracted:
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
        elif expected_type == 'bool':
          if value.lower() == 'true' or value.lower() == 't':
            self.keywords[kwd_lower]['value'] = True
          else:
            self.keywords[kwd_lower]['value'] = False
        elif expected_type == 'str':
          self.keywords[kwd_lower]['value'] = value
        # this keyword has value read in from file, record it
        self.keywords[kwd_lower]['readin'] = True
        self.keywords[kwd_lower]['name'] = kwd

  def extract_block(self, lines):
    line_num = 0

    while line_num < len(lines):
      line = lines[line_num]
      split = line.split()
      block = []
      if split[0].lower() == '%block':
        block_keyword = split[1].lower()
        block.append(line)
        lines.remove(line)
        while line_num < len(lines):
          line = lines[line_num]
          block.append(line)
          lines.remove(line)
          split = line.split()
          if split[0].lower() == '%endblock':
            break
        if block_keyword in self.blocks.keys():
          self.blocks[block_keyword] = block
      else:
        line_num += 1
    return lines


  def echo(self):
    '''Echo the keywords that are read in from file.'''
    print '='*80
    print 'Keywords read from input file:\n'
    for name in self.keywords:
      if self.keywords[name]['readin']:
        print '{:15} {}'.format(self.keywords[name]['name'],
                    self.keywords[name]['value'])

  def get_value(self, key):
    ''' Get the "value" of keywords "key". '''
    return self.keywords[key]['value']

  def get_block(self, key):
    ''' Get a list of input lines block characterized by the "key".'''
    return self.blocks[key]


def parse_arg():
  '''Parse stdin argument.'''
  parser = argparse.ArgumentParser(description='OpenKIM potential model fitting program.')
  parser.add_argument('file_in', type=str, help='Name of input file that contains the '
                          'keywords of the program.')
  args = parser.parse_args()
  return [args.file_in]



# test
if __name__ == '__main__':
  #keys = InputKeywords()
  #keys.read('develop_test/kimfit.in')
  #keys.echo()

  class write_all_keywords(InputKeywords):
    def write(self, fname='./kimfit.log'):
      'Write all keywords to file.'
      with open(fname, 'w') as fout:
        for name in self.keywords:
          fout.write('{}   {}\n'.format(name, self.keywords[name]['value']))

  keys = write_all_keywords()
  keys.read()
  keys.echo()
  keys.write('./kimfit.log')



