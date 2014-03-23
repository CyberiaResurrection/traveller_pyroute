'''
Created on Mar 22, 2014

@author: tjoneslo
'''

class WikiStats(object):
    '''
    classdocs
    '''


    def __init__(self, galaxy, uwp):
        '''
        Constructor
        '''
        self.galaxy = galaxy
        self.uwp  = uwp

    def write_statistics(self):
        self.summary_statistics()
        self.top_summary()
        
    def summary_statistics(self):
        with open ('summary.wiki', 'w+') as f:
            f.write('==Economic Summary==\n')
            f.write('===Statistical Analysis===\n')
            f.write('Ppoulations are in millions, economy and trade in billions.\n')
            f.write('{| class=\"wikitable sortable\"\n!Sector!!X,Y!!Worlds !!Pop!!Economy !!Per Capita!!Trade Volume!! Int. Trade !! Ext. Trade\n')
            for sector in self.galaxy.sectors:
                f.write('|-\n')
                f.write('|[[{0} Sector|{0}]]\n'.format(sector.name))
                f.write('|| {:d},{:d}\n'.format(sector.x, sector.y))
                self.write_stats(f, sector.stats)
                
            if 'Im' in self.galaxy.alg:
                f.write('|-\n')
                f.write('| Imperial Total ||\n')
                imStats = self.galaxy.alg['Im'][1]
                self.write_stats(f, imStats)
            
            f.write('|-\n')
            f.write('| Global Total || \n')
            self.write_stats(f, self.galaxy.stats) 
            f.write('|}\n')
            
    def write_stats(self, f, stats):
        f.write('|align="right"|{:d}\n'.format(stats.number))
        f.write('|align="right"|{:,d}\n'.format(stats.population))
        f.write('|align="right"|{:,d}\n'.format(stats.economy))
        f.write('|align="right"|{:,d}\n'.format(stats.percapita))
        f.write('|align="right"|{:,d}\n'.format(int(stats.trade/1e9)))
        f.write('|align="right"|{:,d}\n'.format(0))

    def top_summary(self):
        with open ('top_summary.wiki', 'w+') as f:
            f.write('==Top level Summary==\n')
            f.write('{|\n')
            f.write('|Systems || {}\n'.format(self.galaxy.stats.number))
            f.write('|-\n')
            f.write('|Population || {:,d} million\n'.format(self.galaxy.stats.population))
            f.write('|-\n')
            f.write('|Gross Total Product || {:,d} billion\n'.format(self.galaxy.stats.economy))
            f.write('|-\n')
            f.write('|Per Capita GSP || Cr {:,d}\n'.format(self.galaxy.stats.percapita))
            f.write('|-\n')
            f.write('|Trade || {:,d} billion\n'.format(int(self.galaxy.stats.trade/1e9)))
            f.write('\n|}\n')
            f.write('===[[Allegiance Code|Allegiance Information]]===\n')
            self.write_allegiances(f,self.galaxy.alg)
            f.write('===Summary Report===\n')
            self.write_uwp_report(f)
        
    def write_allegiances (self,f,alg):
        f.write('{| class=\"wikitable sortable\"\n!Code ||Name||Worlds||Population||GNP\n')
        for code in alg.iterkeys():
            if alg[code][1].number < 10:
                continue
            f.write('|-\n| {} || [[{}]] '.format(code, alg[code][0]))
            stats = alg[code][1]
            f.write('|| {:,d} || {:,d} || {:,d}\n'.format(stats.number, stats.population, stats.economy))
        f.write('|}\n')
            
    
    def write_uwp_report(self,f):
        f.write('{|\n!Component')
        for x in range(18):
            f.write('||{}'.format(baseN(x,18)))
        f.write('||I-Z')
        for name in self.uwp.iterkeys():
            f.write('\n|- \n| {}'.format(name))
            for x in range(18):
                index = baseN(x,18)
                stats = self.uwp.get(name, {})
                value = stats.get(index, '0')
                value = value.number if value != '0' else '0'
                f.write('||{}'.format(value))
            if self.uwp[name].keys() in ['X']:
                stats = self.uwp.get(name, {})
                value = stats.get(index, '0')
                value = value.number if value != '0' else '0'
                f.write ('||{}'.format(value))
            else:
                f.write('||0')
        f.write('\n|}')
        
        
def baseN(num,b,numerals="0123456789ABCDEFGHijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])        