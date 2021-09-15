import re

def _perc_calc(prob, lf):
    # contains 'percent symbol'
    if '%' in prob:
        per = float(prob[:-1]) / 100 * float(lf)
        return per
    else:
        return prob


def _solve(p):

    """Does the real evaluation
        """

    operators = ('-', '+', '*', '/')
    print(p, ': has come into evaluate')
    for operator in operators:
        if operator in p:
            splits = p.split(operator, 1)
            print('splits: ', splits)
            # check if string contains
            # only numbers
            left_sp = [o for o in operators if o in splits[0]]
            right_sp = [p for p in operators if p in splits[1]]
            if operator == '-':
                if left_sp:
                    left = _solve(splits[0])
                else:
                    left = splits[0]

                if right_sp:
                    right = _solve(splits[1])
                else:
                    # contains 'percent symbol'
                    if '%' in splits[1]:
                        print('except: ', splits[:-2])
                    right = splits[1]

                solute = float(left) - float(right)
                return solute

            elif operator == '+':
                print('ss: ', len(splits[0]))
                if left_sp:
                    left = _solve(splits[0])
                else:
                    print('ff')
                    left = splits[0]

                if right_sp:
                    print('gere')
                    right = _solve(splits[1])
                else:
                    # contains 'percent symbol'
                    if '%' in splits[1]:
                        per = float(splits[1][:-1]) / 100 * float(left)
                        right = per
                    else:
                        right = splits[1]

                solute = float(left) + float(right)
                return solute

            elif operator == '*':
                if left_sp:
                    left = _solve(splits[0])
                else:
                    left = splits[0]

                if right_sp:
                    right = _solve(splits[1])
                else:
                    # contains 'percent symbol'
                    if '%' in splits[1]:
                        print('except: ', splits[:-2])
                    right = splits[1]

                solute = float(left) * float(right)
                return solute

            elif operator == '/':
                if left_sp:
                    left = _solve(splits[0])
                else:
                    left = splits[0]

                if right_sp:
                    right = _solve(splits[1])
                else:
                    # contains 'percent symbol'
                    if '%' in splits[1]:
                        print('except: ', splits[:-2])
                    right = splits[1]

                solute = float(left) / float(right)
                return solute

            else:
                pass
        else:
            pass


prob = '10+20%'
operates = ('-', '+', '/', '*')
lists = re.findall('[0-9]+%', prob)
print(lists)
ll = [o+each_per for o in operates for each_per in lists if o+each_per in prob]
sign = ll[0][0]
print(sign)
print(ll)
spls = prob.split(ll[0])
print(spls)
# ll = [n for o in operates if o in prob]


