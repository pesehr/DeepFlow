import sys
import xml.etree.ElementTree as ET
import math


class XMLParser:
    def __init__(self, address):
        self.address = address

    def read_txt(self):
        f = open(self.address, 'r')
        a = []
        b = []
        lines = f.readlines()
        for k in range(0, len(lines)):
            l = lines[k]
            if l != '\n':
                d = l.strip().split(";")
                r = d[0].strip().split(" ")
                a.append(
                    [[float(r[i].split(",")[0])]
                    # [[float(r[i].split(",")[0]) - float(r[i - 1].split(",")[0]) if i > 0 else 0]
                     for i in
                     range(0, len(r))])
                b.append(float(d[1]))
        return a, b
