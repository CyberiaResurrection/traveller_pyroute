"""
Created on Dec 27, 2023

@author: CyberiaResurrection
"""
import re

from lark import Transformer, Token


class StarlineTransformer(Transformer):

    strip_list = [
        'pbg',
        'nobles',
        'zone',
        'allegiance',
        'base',
        'residual'
    ]

    def __init__(self, visit_tokens: bool = True, raw=None):
        super().__init__(visit_tokens)
        self.raw = raw

    def starline(self, args):
        tradelen = sum([len(item) for item in args[2]]) + len(args[2]) - 1
        if 16 < tradelen and 3 <= len(args[2]) and 1 == len(args[3]) and '' == args[3][0].value.strip():  # Square up overspilled trade codes
            if '' == args[4][0].value and '' != args[5][0].value and '' == args[6][0].value:
                move_fwd = 3 == len(args[5][0].value)  # Will base code still make sense as PBG?
                move_rev = 3 == len(args[7][2][0].value)  # Will allegiance code still make sense as PBG?
                assert not (move_rev and move_fwd), "Move flags should be disjoint"
                if move_fwd:
                    last = args[2][-1]
                    mid = args[2][-2]
                    args[6][0].value = args[5][0].value
                    args[5][0].value = last
                    args[4][0].value = mid
                    args[2] = args[2][:-2]
                elif move_rev:
                    pass
        if '*' != args[5][0].value:
            if '' == args[4][0].value and '' != args[5][0].value:
                args[4][0].value = args[5][0].value
                args[5][0].value = args[6][0].value
            elif '' == args[6][0].value and '' != args[5][0].value:
                args[6][0].value = args[5][0].value
                args[5][0].value = args[4][0].value
                args[4][0].value = ''
        if 8 == len(args):  # If there's no residual argument
            tailend = args[7][2][0].value
            lenlast = min(4, len(tailend))
            counter = 0
            while counter < lenlast and (tailend[counter].isalnum() or '-' == tailend[counter] or '?' == tailend[counter]):
                if counter < lenlast:
                    counter += 1
            if counter < min(4, lenlast):  # if the allegiance overspills, move the overspill into the residual
                overrun = tailend[counter:]
                tailend = tailend[:counter]
                args[7][2][0].value = tailend
                newbie = Token('__ANON_14', overrun)
                args.append([newbie])
        return args

    def position(self, args):
        args[0].value = args[0].value.strip()
        return args

    def starname(self, args):
        args[0].value = args[0].value.strip()
        return args

    def trade(self, args):
        trimmed = []
        for item in args:
            trimmed.append(item.value.strip())
        return trimmed

    def extensions(self, args):
        if 1 == len(args):
            return args
        return args

    def nobles(self, args):
        args[0].value = args[0].value.strip()
        return args

    def base(self, args):
        args[0].value = args[0].value.strip()
        return args

    def zone(self, args):
        args[0].value = args[0].value.strip()
        return args

    def pbg(self, args):
        args[0].value = args[0].value.strip()
        return args

    def worlds(self, args):
        raw = args[0].value
        if 1 < len(raw):
            raw = raw.strip()
        args[0].value = raw
        return args

    def allegiance(self, args):
        args[0].value = args[0].value.strip()
        return args

    def world_alg(self, args):
        return args

    def residual(self, args):
        args[0].value = args[0].value.strip()
        return args

    def starname_transform(self, starname):
        bitz = [item for item in starname.split(' ') if 0 < len(item)]
        uwp = bitz[-1]
        bitz = bitz[:-1]
        return ' '.join(bitz), uwp

    def trade_transform(self, trade):
        codes = []

        for kid in trade:
            codes.append(kid)

        return ' '.join(codes)

    def extensions_transform(self, extensions):
        if 1 == len(extensions):  # Fallback no-extensions
            return None, None, None
        data = {}
        for kid in extensions:
            if isinstance(kid, Token):
                val = str(kid.type)
                data[val] = kid.value
            else:
                val = str(kid.data)
                data[val] = kid.children[0].value
            data[val] = self.boil_down_double_spaces(data[val])

        return data['ix'], data['ex'], data['cx']

    def world_alg_transform(self, world_alg):
        return world_alg[0][0].value, world_alg[1][0].value, world_alg[2][0].value

    def transform(self, tree):
        tree = self._preprocess_trade_and_extensions(tree)
        tree = self._preprocess_extensions_and_nbz(tree)
        tree = self._preprocess_trade_and_nbz(tree)
        tree = self._transform_tree(tree)
        parsed = {'ix': None, 'ex': None, 'cx': None, 'residual': ''}

        parsed['position'] = tree[0][0].value
        parsed['name'], parsed['uwp'] = self.starname_transform(tree[1][0].value)
        parsed['trade'] = self.trade_transform(tree[2])
        parsed['ix'], parsed['ex'], parsed['cx'] = self.extensions_transform(tree[3])
        parsed['nobles'] = tree[4][0].value
        parsed['base'] = tree[5][0].value
        parsed['zone'] = tree[6][0].value
        parsed['pbg'], parsed['worlds'], parsed['allegiance'] = self.world_alg_transform(tree[7])
        if 9 == len(tree):
            parsed['residual'] = tree[8][0].value

        parsed = self._square_up_parsed(parsed)
        self.trim_raw_string(parsed)
        rawbitz = self._trim_raw_bitz(parsed)
        parsed = self._square_up_parsed_zero(rawbitz[0], parsed)
        parsed = self._square_up_parsed_one(rawbitz[1], parsed)
        parsed = self._square_up_allegiance_overflow(parsed)

        no_extensions = parsed['ix'] is None and parsed['ex'] is None and parsed['cx'] is None
        ex_ix = parsed['ix'] if parsed['ix'] is not None else ' '
        ex_ix = self.boil_down_double_spaces(ex_ix)
        extensions = (ex_ix) + ' ' + \
                     (parsed['ex'] if parsed['ex'] is not None else ' ') + ' ' + \
                     (parsed['cx'] if parsed['cx'] is not None else ' ')

        spacer = ' ' if no_extensions else None
        if no_extensions:
            extensions = ''

        # Currently aiming to drop-in replace the starline regex output
        data = [parsed['position'], parsed['name'], parsed['uwp'], parsed['trade'], extensions, parsed['ix'], parsed['ex'], parsed['cx'], spacer, spacer, spacer, parsed['nobles'], parsed['base'], parsed['zone'], parsed['pbg'], parsed['worlds'], parsed['allegiance'], parsed['residual']]

        return data

    def _preprocess_trade_and_extensions(self, tree):
        trade = tree.children[2]
        extensions = tree.children[3]
        ix_reg = '\{ *[+-]?[0-6] ?\}'

        # If trade has importance-extension child, we need to fix it
        counter = -1
        ix_found = False
        for kid in trade.children:
            counter += 1
            ix_match = re.match(ix_reg, kid.value)
            if ix_match is not None:
                ix_found = True
                break

        if not ix_found:
            return tree

        bitz = trade.children[counter:]
        bitz[0].type = 'ix'
        bitz[1].type = 'ex'
        bitz[2].type = 'cx'
        trade.children = trade.children[:counter]
        extensions.children.extend(bitz)

        return tree

    def _preprocess_extensions_and_nbz(self, tree):
        extensions = tree.children[3]
        if 5 > len(extensions.children):  # Nothing to move around, bail out
            return tree

        nobles = tree.children[4]
        # base = tree.children[5]
        zone = tree.children[6]

        nobles.children[0].value = extensions.children[4].value
        if 5 < len(extensions.children):
            zone.children[0].value = extensions.children[5].value

        return tree

    def _preprocess_trade_and_nbz(self, tree):
        trade = tree.children[2]
        tradelen = sum([len(item.value) for item in trade.children]) + (len(trade.children) - 1)
        if 17 > tradelen:
            return tree

        nobles = tree.children[4]
        base = tree.children[5]
        zone = tree.children[6]
        world_alg = tree.children[7]
        pbg = world_alg.children[0]
        worlds = world_alg.children[1]

        if 2 < len(base.children[0].value):
            if 2 < len(trade.children):
                if zone.children[0].value.strip() == worlds.children[0].value.strip():
                    worlds.children[0].value = pbg.children[0].value
                    pbg.children[0].value = base.children[0].value
                relocate = trade.children[-2:]
                trade.children = trade.children[:-2]
                zone.children[0].value = nobles.children[0].value
                base.children[0].value = relocate[1].value
                nobles.children[0].value = relocate[0].value
        elif 3 == len(pbg.children[0].value):
            if 2 < len(trade.children):
                relocate = [item for item in trade.children[-2:] if 1 == len(item)]
                if 2 == len(relocate):
                    trade.children = trade.children[:-2]
                    zone.children[0].value = nobles.children[0].value
                    base.children[0].value = relocate[1].value
                    nobles.children[0].value = relocate[0].value
                elif '*' == trade.children[-1]:
                    relocate = trade.children[-2:]
                    trade.children = trade.children[:-2]
                    zone.children[0].value = nobles.children[0].value
                    base.children[0].value = relocate[1].value
                    nobles.children[0].value = relocate[0].value
            elif 2 == len(trade.children) and '' == nobles.children[0].value:
                relocate = trade.children[-1:]
                trade.children = trade.children[:-1]
                nobles.children[0].value = relocate[0].value

        return tree

    def _square_up_parsed(self, parsed):
        parsed = self._square_up_parsed_trade_codes(parsed)
        if ' ' != parsed['nobles'] and 0 < len(parsed['nobles']) and '' == parsed['base'] and '' == parsed['zone']:
            parsed['base'] = parsed['pbg']
            parsed['zone'] = parsed['worlds']
            parsed['pbg'] = parsed['allegiance']
            parsed['worlds'] = ' '
            parsed['allegiance'] = parsed['residual']
            parsed['residual'] = ''

        return parsed

    def _square_up_parsed_trade_codes(self, parsed):
        bitz = parsed['trade'].split(' ')
        if 5 == len(bitz) and parsed['ix'] is None and parsed['ex'] is None and parsed['cx'] is None and '' == parsed['nobles'].strip():
            parsed['nobles'] = bitz[-1]
            parsed['cx'] = bitz[-2]
            parsed['ex'] = bitz[-3]
            parsed['ix'] = bitz[-4]
            bitz = bitz[:-4]
        elif '*' in parsed['trade'] and '' != parsed['nobles'] and '' == parsed['base'] and '' == parsed['zone']:
            parsed['zone'] = parsed['nobles']
            parsed['nobles'] = ''
            parsed['base'] = '*'
            bitz = [item for item in bitz if '*' != item]
        elif 15 < len(parsed['trade']) and ' ' in parsed['trade'] and '' == parsed['nobles'] and '' != parsed[
            'base'] and '' != parsed['zone']:
            parsed['nobles'] = bitz[-1]
            bitz = bitz[:-1]

        parsed['trade'] = ' '.join(bitz)
        return parsed

    def trim_raw_string(self, tree):
        assert self.raw is not None, "Raw string not supplied before trimming"
        strip_list = ['position', 'name', 'uwp', 'trade', 'ix', 'ex', 'cx']

        for dataval in strip_list:
            if dataval not in tree:
                continue
            rawval = tree[dataval]
            if rawval is not None:
                self.raw = self.raw.replace(rawval, '', 1)

    def _square_up_parsed_zero(self, rawstring, parsed):
        bitz = [item for item in rawstring.split(' ') if '' != item]
        if 3 == len(bitz) and bitz[0] == parsed['nobles'] and bitz[1] == parsed['base'] and bitz[2] == parsed['zone']:
            return parsed
        if 3 == len(bitz):
            parsed['nobles'] = bitz[0]
            parsed['base'] = bitz[1]
            parsed['zone'] = bitz[2]
        if 2 == len(bitz) and '*' != parsed['base']:
            if not rawstring.endswith('   '):
                parsed['nobles'] = ''
                parsed['base'] = bitz[0]
                parsed['zone'] = bitz[1]
            else:
                parsed['nobles'] = bitz[0]
                parsed['base'] = bitz[1]
                parsed['zone'] = ''
        return parsed

    def _square_up_parsed_one(self, rawstring, parsed):
        rawtrim = rawstring.lstrip()
        rawbitz = rawtrim.split(' ')
        trimbitz = [item for item in rawbitz if '' != item]
        if 3 < len(trimbitz):
            parsed['worlds'] = trimbitz[0]
            parsed['allegiance'] = trimbitz[1]
            parsed['residual'] = ' '.join(trimbitz[2:])
        elif 3 == len(trimbitz):
            if trimbitz[0].isdigit():
                parsed['worlds'] = trimbitz[0]
                parsed['allegiance'] = trimbitz[1]
                parsed['residual'] = trimbitz[2]
            else:
                parsed['worlds'] = ' '
                parsed['allegiance'] = trimbitz[0]
                parsed['residual'] = trimbitz[1] + ' ' + trimbitz[2]
        elif len(rawtrim) + 3 <= len(rawstring):  # We don't have three matches, need to figure out how they drop in
            alg = trimbitz[0]
            rawtrim = rawtrim.replace(alg, '', 1)

            if 2 == len(trimbitz):
                allegiance = trimbitz[1]
                rawtrim = rawtrim.replace(allegiance, '', 1)
                if alg.isdigit() and 5 > len(alg):  # if first trimbit fits in worlds field, stick it there
                    parsed['worlds'] = alg
                    parsed['allegiance'] = allegiance
                    parsed['residual'] = rawtrim.strip()
                else:
                    parsed['worlds'] = ' '
                    parsed['allegiance'] = alg
                    parsed['residual'] = allegiance

            elif 1 == len(alg):  # Allegiance codes can't be single-char, so we actually have a worlds field
                parsed['worlds'] = alg
                parsed['allegiance'] = rawtrim.strip()
            else:
                parsed['worlds'] = ' '
                parsed['allegiance'] = alg
                parsed['residual'] = rawtrim.strip()
        else:  # Assume worlds field is _not_ blank
            if ' ' == parsed['worlds'] and 2 == len(trimbitz):  # if worlds field has been _parsed_ as blank, need to move allegiance and residual up one
                parsed['worlds'] = trimbitz[0]
                parsed['allegiance'] = trimbitz[1]
                parsed['residual'] = ''

        if '' == parsed['allegiance'].strip() and '' != parsed['residual']:  # Allegiance _must_ be filled, residual is optional, so switch them back if they're transposed
            parsed['allegiance'] = parsed['residual']
            parsed['residual'] = ''

        return parsed

    def _square_up_allegiance_overflow(self, parsed):
        alleg = parsed['allegiance']
        if '----' == alleg or '--' == alleg:
            return parsed

        if alleg.startswith('----') and 4 <= len(alleg):
            parsed['allegiance'] = '----'
            parsed['residual'] = alleg[4:] + parsed['residual']
        elif alleg.startswith('--') and 2 <= len(alleg):
            parsed['allegiance'] = '--'
            parsed['residual'] = alleg[2:] + parsed['residual']
        else:
            counter = 0
            while counter < len(alleg) and (alleg[counter].isalnum() or '-' == alleg[counter] or '?' == alleg[counter]) and 4 > counter:
                counter += 1
            if counter < len(alleg):
                spacer = ' ' if parsed['residual'] != '' else ''
                parsed['allegiance'] = alleg[:counter]
                parsed['residual'] = alleg[counter:] + spacer + parsed['residual']
        return parsed

    def _trim_raw_bitz(self, parsed):
        pbg = ' ' + parsed['pbg'] + ' '
        rawbitz = self.raw.split(pbg)
        oldlen = len(rawbitz)
        if 1 == oldlen:
            rawbitz.append('')
        if 2 < oldlen:
            collide = self._check_raw_collision(parsed)
            first = pbg.join(rawbitz[:-1]) if not collide else pbg.join(rawbitz[:-2])
            second = rawbitz[-1] if not collide else pbg.join(rawbitz[-2:])
            repack = []
            repack.append(first)
            repack.append(second)
            rawbitz = repack
        rawbitz[0] += ' '
        rawbitz[1] = ' ' + rawbitz[1]

        return rawbitz

    def _check_raw_collision(self, parsed):
        if parsed['pbg'] == parsed['worlds']:
            return True
        if parsed['pbg'] == parsed['allegiance']:
            return True
        if parsed['pbg'] == parsed['residual']:
            return True
        return False

    @staticmethod
    def boil_down_double_spaces(dubbel):
        return " ".join(dubbel.split())
