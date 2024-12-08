import decimal
from AevisTemp import *

test_data = [
                [
                    [decimal.Decimal('1') , decimal.Decimal('2') , decimal.Decimal('3')],
                    decimal.Decimal('1')
                ],
                [
                    [decimal.Decimal('2') , decimal.Decimal('3'), decimal.Decimal('4')],
                    decimal.Decimal('2')
                ],
                [
                    [decimal.Decimal('3') , decimal.Decimal('4') , decimal.Decimal('5')],
                    decimal.Decimal('3')
                ]         
             ]

nn = StackedLSTM()

nn.learn(test_data)