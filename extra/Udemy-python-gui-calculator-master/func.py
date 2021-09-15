import re
import threading
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class Calculator(QObject):

    def __init__(self):
        QObject.__init__(self)
        self.operators = ('-', '+', '/', '*')

    evaluated = pyqtSignal(str, arguments=["_compute"])

    @pyqtSlot(str)
    def compute(self, stat):
        comp_thread = threading.Thread(target=self._compute, args=[stat])
        comp_thread.daemon = True
        comp_thread.start()

    def _solve(self, pb):
        """Does the real evaluation
        """

        print(pb, ': has come into solve')
        if '%' in pb:
            p = self.evaluate(pb)
        else:
            p = pb
        for operator in self.operators:
            if operator in p:
                splits = p.split(operator, 1)
                print('splits: ', splits)
                # check if string contains
                # only numbers
                left_sp = [o for o in self.operators if o in splits[0]]
                right_sp = [p for p in self.operators if p in splits[1]]
                if operator == '-':
                    if left_sp:
                        left = self._solve(splits[0])
                    else:
                        left = splits[0]

                    if right_sp:
                        right = self._solve(splits[1])
                    else:
                        right = self._perc_calc(splits[1], left)

                    if not left:
                        left = '0.0'
                    if not right:
                        right = '0.0'
                    solute = float(left) - float(right)
                    return solute

                elif operator == '+':
                    print('ss: ', len(splits[0]))
                    if left_sp:
                        left = self._solve(splits[0])
                    else:
                        print('ff')
                        left = splits[0]

                    if right_sp:
                        print('gere')
                        right = self._solve(splits[1])
                    else:
                        print('wee')
                        right = self._perc_calc(splits[1], left)

                    if not left:
                        left = '0.0'
                    if not right:
                        right = '0.0'
                    solute = float(left) + float(right)
                    return solute

                elif operator == '*':
                    if left_sp:
                        left = self._solve(splits[0])
                    else:
                        left = splits[0]

                    if right_sp:
                        right = self._solve(splits[1])
                    else:
                        right = self._perc_calc(splits[1], left)

                    if not left:
                        left = '1.0'
                    if not right:
                        right = '1.0'

                    solute = float(left) * float(right)
                    return solute

                elif operator == '/':
                    if left_sp:
                        left = self._solve(splits[0])
                    else:
                        left = splits[0]

                    if right_sp:
                        right = self._solve(splits[1])
                    else:
                        right = self._perc_calc(splits[1], left)

                    if not left:
                        if right:
                            return float(right)
                        else:
                            # might be bad for multiplication and division
                            return 0.0
                    if not right:
                        right = '1.0'

                    solute = float(left) / float(right)
                    return solute

                else:
                    pass

        else:
            # contains no operator
            print('contains no operator')
            if p:
                return float(p)
            else:
                # might cause trouble for multiplication and division
                return 0.0

    def _perc_calc(self, prob, lf):
        # If percentage, calculate percentage
        # return the answer
        if '%' in prob:
            per = float(prob[:-1]) / 100 * float(lf)
            return per
        else:
            return prob

    def evaluate(self, s):
        """Evaluate the problem and return a solution
        """

        # Run to evalute the statement

        # if contains percentage
        if '%' in s:
            percent = re.findall('[0-9]+%', s)
            o_per = [o+per for o in self.operators for per in percent if o+per in s]
            # e.g ['+25%']
            # will be 25
            real_percent = o_per[0][1:-1]
            sign = o_per[0][0]
            prob_split = s.split(o_per[0])
            left_p = prob_split[0]
            right_p = prob_split[1]
            if left_p:
                left_sol = self._solve(left_p)
                print('left_sol', left_sol)
            else:
                # percent doesn't act on any number
                return 0

            if right_p:
                pass
            else:
                right_p = ''

            # get the percentage value
            percent_value = float(left_sol) * float(real_percent) / 100
            # send in the percent with the sign for the final calculation
            statement = str(left_sol) + sign + str(percent_value) + right_p
            print('statement: ', statement)
            result = self._solve(statement)
            print('result: ', result)

        # doesn't contain percentage
        else:
            result = self._solve(s)

        if not result:
            result = 0.0
        return str(result)

    def _compute(self, problem):
        # call evaluate to do the evaluation first
        result = self._solve(problem)

        # find if it can be converted to an integer
        diff = result - int(result)
        if diff == 0.0:
            final = int(result)
        else:
            final = result
        self.evaluated.emit(str(final))
