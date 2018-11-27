#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


class Tree(object):

    def __init__(self):
        self.dirCount = 0
        self.stack_num_all = []
        self.stack_num_current = []
        self.string = ''
        self.started = False

    def count(self, absolute):
        if os.path.isfile(absolute) and absolute.endswith('.xyz'):
            return 1
        else:
            return 0

    def summary(self):
        s = '\n' + '='*80 + '\n'
        s += '                           KLIFF Dataset Count\n\n'
        s += 'Notation: "──dir_name (a/b)"\n'
        s += 'a: number of .xyz files in the directory "dir_name"\n'
        s += 'b: number of .xyz files in the directory "dir_name" and its subdirectories\n'
        s += '\n'
        s += self.string
        s += '\n' + '='*80 + '\n'

        return s

    def walk(self, directory, prefix=''):

        num_current = 0
        num_all = 0

        if not self.started:
            self.string += '{}{} (rpls_num_current_{}/rpls_num_all_{})\n'.format(
                prefix, directory, self.dirCount, self.dirCount)
            self.stack_num_all.append('rpls_num_all_{}'.format(self.dirCount))
            self.stack_num_current.append(
                'rpls_num_current_{}'.format(self.dirCount))
            self.dirCount += 1
            self.started = True

        filepaths = os.listdir(directory)
        for index, path in enumerate(filepaths):

            if path[0] == '.':
                continue

            absolute = os.path.join(directory, path)
            x = self.count(absolute)
            num_current += x
            num_all += x

            if index == len(filepaths) - 1:
                connector = '└──'
                prefix_new = prefix + '    '
            else:
                connector = '├──'
                prefix_new = prefix + '│   '

            if os.path.isdir(absolute):
                self.string += '{}{}{} (rpls_num_current_{}/rpls_num_all_{})\n'.format(
                    prefix, connector, path, self.dirCount, self.dirCount)
                self.stack_num_all.append(
                    'rpls_num_all_{}'.format(self.dirCount))
                self.stack_num_current.append(
                    'rpls_num_current_{}'.format(self.dirCount))
                self.dirCount += 1
                sub_current, sub_all = self.walk(absolute, prefix_new)
                num_all += sub_all

        self.string = self.string.replace(
            self.stack_num_all.pop(), str(num_all))
        self.string = self.string.replace(
            self.stack_num_current.pop(), str(num_current))

        return num_current, num_all


def dataset_count(directory):
    tree = Tree()
    tree.walk(directory)
    print(tree.summary())


if __name__ == '__main__':

    directory = "."
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    tree = Tree()
    tree.walk(directory)
    print(tree.summary())
