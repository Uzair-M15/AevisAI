import math
from random import random
import decimal

#region Vectors and Activations

class Vector:
    def __init__(self , dimensions = 2 , values = [ 0  , 0] , dec = True):
        decimal.getcontext().prec = 99
        self.dec = dec

        self.dim = dimensions

        if values == [0 , 0] and dimensions != 2 :
            values = [i-i for i in range(dimensions)]
        elif len(values) != self.dim :
            self.dim = len(values)

        self.components = []
        for i in range(self.dim):
            self.components.append(decimal.Decimal(str(values[i])))
    
    def set_component(self , index , value):
        self.components[index] = value
    
    def get_component(self , index):
        return self.components[index]
    
    def resize(self , new_dim):
        if new_dim > self.dim :
            self.components = self.components[::] + [i-i for i in range(new_dim - self.dim)]
        elif new_dim < self.dim :
            self.components = self.components[0:(new_dim):1] 
        self.dim = new_dim
    
    def addition(left , right):
        if left.dim > right.dim :
            right.resize(left.dim)
        elif left.dim < right.dim :
            left.resize(right.dim)
        
        vec = Vector(left.dim)
        vec.components = [left.components[i] + right.components[i] for i in range(len(left.components))]
        return vec
    
    def subtraction(left , right):
        if left.dim > right.dim :
            right.resize(left.dim)
        elif left.dim < right.dim :
            left.resize(right.dim)
        
        vec = Vector(left.dim)
        vec.components = [left.components[i] - right.components[i] for i in range(len(left.components))]
        return vec

    def __add__(self , other):
        vec = Vector(self.dim)
        for i in range(self.dim) :
            vec.set_component(i , self.get_component(i) + other)
    def __pow__(self , exponent : float):
        vec = Vector(self.dim)
        for i in range(self.dim):
            vec.set_component(i , self.get_component(i)**exponent)
        return vec
    def __truediv__(self , divisor : float):
        dividends = Vector(self.dim)
        for i in range(self.dim):
            dividends.set_component(i , self.get_component(i) + divisor)
        return dividends
    def __floordiv__(self , divisor : float):
        dividends = Vector(self.dim)
        for i in range(self.dim) :
            dividends.set_component(i , self.get_component(i)//divisor)
        return dividends
    def __sub__(self , other ):
        vec = Vector(self.dim)
        for i in range(self.dim) :
            vec.set_component( i , self.get_component(i)-other)
        return vec

    def __radd__(self , other):
        vec = Vector(self.dim)
        for i in range(self.dim) :
            vec.set_component(i , self.get_component + other)
        return vec
    def __rpow__(self , other):
        vec = Vector(self.dim)
        for i in range(self.dim):
            vec.set_component(i , other**self.get_component(i))
        return vec
    def __rtruediv__(self , other):
        vec = Vector(self.dim)
        for i in range(self.dim) :
            vec.set_component(i , other/self.get_component(i))
        return vec
    def __rfloordiv__(self,other):
        vec = Vector(self.dim)
        for i in range(self.dim): 
            vec.set_component(i , other//self.get_component(i))
        return vec
    def __rsub__(self , other):
        vec = Vector(self.dim)
        for i in range(self.dim) :
            vec.set_component(i , other - self.get_component(i))
        return vec

class Activation:
    def __init__(self , type = 'none'):
        self.type = type
    def sigmoid(self):
        self.type = 'sigmoid'
    def tanh(self):
        self.type = 'tanh'
    def __call__(self , x):
        if self.type == 'sigmoid':
            return (decimal.Decimal('1')/(decimal.Decimal('1')+(decimal.Decimal(str(math.e))**-x)))
        if self.type == 'tanh':
            return (decimal.Decimal('2')/(decimal.Decimal('1')+decimal.Decimal(str(math.e))**((-2)*x)))-decimal.Decimal('1')
        if self.type == 'none':
            return x

#endregion

#region Primitives

class Connection:
    def __init__(self , weight = random()):
        self.weight = weight

    def __call__(self , x):
        return self.weight*x

class Neuron :
    def __init__(self , bias = random() , weight = random() , activation = Activation()):
        self.bias = decimal.Decimal(str(bias))
        self.activation = activation
        self.weight = decimal.Decimal(str(weight))
    
    def __call__(self , x):
        return self.activation(self.bias + (x*self.weight))

class lstmCell:
    def __init__(self  ,
                 forget_input_weight = random() , 
                 forget_hidden_weight = random() , 
                 forget_bias = random() , 
                 input_input_weight = random() , 
                 input_hidden_weight = random() , 
                 input_bias = random() , 
                 candidate_input_weight = random() , 
                 candidate_hidden_weight = random() , 
                 candidate_bias = random() ,
                 output_hidden_weight = random(),
                 output_input_weight = random(),
                 output_bias = random() ,
                 amplitude = 1
                 ):
        
        self.sigmoid = Activation('sigmoid')
        self.tanh = Activation('tanh')
        
        self.forget_input_weight =  decimal.Decimal(str(forget_input_weight))
        self.forget_hidden_weight = decimal.Decimal(str(forget_hidden_weight))
        self.forget_bias = decimal.Decimal(str(forget_bias))
        
        self.input_input_weight = decimal.Decimal(str(input_input_weight))
        self.input_hidden_weight = decimal.Decimal(str(input_hidden_weight))
        self.input_bias = decimal.Decimal(str(input_bias))

        self.candidate_input_weight = decimal.Decimal(str(candidate_input_weight))
        self.candidate_hidden_weight = decimal.Decimal(str(candidate_hidden_weight))
        self.candidate_bias = decimal.Decimal(str(candidate_bias))

        self.output_input_weight = decimal.Decimal(str(output_input_weight))
        self.output_hidden_weight = decimal.Decimal(str(output_hidden_weight))
        self.output_bias = decimal.Decimal(str(output_bias))

        self.hidden_state = decimal.Decimal(str(0))
        self.cell_state = decimal.Decimal(str(0))
        self.amplitude = amplitude
    
    def __call__(self , x):
        # forget gate
        c = self.cell_state*(self.sigmoid(self.forget_bias + ((self.hidden_state*self.forget_hidden_weight) + (x*self.forget_input_weight)) ))
        self.previous_cell_state = self.cell_state
        
        #input gate
        input_out = (self.sigmoid( ( (self.input_hidden_weight*self.hidden_state)     + (self.input_input_weight*x))     + self.input_bias )     * 
                     self.tanh(    ( (self.candidate_hidden_weight*self.hidden_state) + (self.candidate_input_weight*x)) + self.candidate_bias ))
        
        c = c + input_out
        self.cell_state = c

        #output
        o = self.sigmoid( (self.output_hidden_weight*self.hidden_state) + (self.output_input_weight*x))

        final_output = o*self.amplitude*self.tanh(c)
        self.hidden_state = final_output

        return final_output

    def learn(self , x):

        cost = 0
        predictions = []
        hidden_states = []
        cell_states = []
        outputs = []
        inputs = []

        #Forward pass
        print("Forward Pass: ")
        for io_pair in x :        
            input_data = io_pair[0]
            output_data = io_pair[1]

            hidden_states.append(self.hidden_state)
            print("")
            print("     Hidden state:" , hidden_states[len(hidden_states)-1])
            cell_states.append(self.cell_state)
            print("     Cell State:", cell_states[len(cell_states)-1])

            for inp in input_data :
                o = self.__call__(inp)
            predictions.append(o)            
            
            print("     Prediction:",predictions[len(predictions)-1])
            outputs.append(output_data)
            print("     Data:" , outputs[len(outputs)-1])
            inputs.append(input_data)
        
        cell_states.append(self.cell_state)
        hidden_states.append(self.hidden_state)

        print("")
        print("Backwards Pass:")
        for i in reversed(range(len(predictions))) :
            cost = decimal.Decimal('0.5')*((outputs[i] - predictions[i])**2)
            print("")
            print("     Cost:" , cost)
            accuracy = ((predictions[i])/self.amplitude)/(output_data)
            print("     Accuracy:",accuracy)
            learning_rate = decimal.Decimal('0.5')*(1-accuracy)
            print("     Learning Rate:" , learning_rate)

            s = self.candidate_bias + (self.candidate_hidden_weight* hidden_states[i]) + (self.candidate_input_weight*inputs[i][len(inputs[i])-1])
            t = self.input_bias + (self.input_hidden_weight*hidden_states[i]) + (self.input_input_weight *inputs[i][len(inputs[i])-1])
            u = self.forget_bias + (self.forget_hidden_weight*hidden_states[i]) + (self.forget_input_weight*inputs[i][len(inputs[i])-1])
            v = self.output_bias + (self.output_hidden_weight*hidden_states[i]) + (self.output_input_weight*inputs[i][len(inputs[i])-1])

            I_g = self.tanh(s)*self.sigmoid(t)
            F_g = cell_states[i] * self.sigmoid(u)
            O_g = self.sigmoid(v)
            o = self.amplitude*self.tanh(I_g + F_g)*O_g

            grad_bf = -(outputs[i]-o)*((1-((self.tanh(I_g + F_g)**2))**2))*O_g*(cell_states[i]*(self.sigmoid(u)*(1-self.sigmoid(u))))*self.amplitude
            grad_wfh = grad_bf*hidden_states[i]
            grad_wfi = grad_bf*inputs[i][len(inputs[i])-1]

            grad_bi = -(outputs[i]-o)*((1-((self.tanh(I_g + F_g)**2))**2))*O_g*(self.sigmoid(t)*(1-self.sigmoid(t)))*self.tanh(s)*self.amplitude
            grad_wih = grad_bi*hidden_states[i]
            grad_wii = grad_bi*inputs[i][len(inputs[i])-1]
            
            grad_bc = -(outputs[i]-o)*((1-((self.tanh(I_g + F_g)**2))**2))*O_g*self.sigmoid(t)*(1-(self.sigmoid(t))**2)*self.amplitude
            grad_wch = grad_bc*hidden_states[i]
            grad_wci = grad_bc*inputs[i][len(inputs[i])-1]

            grad_bo = -(outputs[i]-o)*(self.tanh(I_g + F_g))*(self.sigmoid(v)*(1-self.sigmoid(v)))*self.amplitude
            grad_woh = grad_bo*hidden_states[i]
            grad_woi = grad_bo*inputs[i][len(inputs[i])-1]

            self.forget_bias =self.forget_bias + (learning_rate*grad_bf)
            self.forget_hidden_weight = self.forget_hidden_weight + (learning_rate*grad_wfh)
            self.forget_input_weight =self.forget_input_weight + (learning_rate*grad_wfi)
            self.input_bias =self.input_bias + (learning_rate*grad_bi)
            self.input_hidden_weight = self.input_hidden_weight + (learning_rate*grad_wih)
            self.input_input_weight = self.input_input_weight + (learning_rate*grad_wii)
            self.candidate_bias = self.candidate_bias + (learning_rate*grad_bc)
            self.candidate_hidden_weight = self.candidate_hidden_weight + (learning_rate*grad_wch)
            self.candidate_input_weight = self.candidate_input_weight + (learning_rate*grad_wci)
            self.output_bias = self.output_bias + (learning_rate*self.grad_bo)
            self.output_hidden_weight =self.output_hidden_weight + (learning_rate*grad_woh)
            self.output_input_weight = self.output_input_weight+ (learning_rate*grad_woi)        


#endregion

#region Models
class StackedLSTM :
    def __init__(self , stack_size = 3 , weights = [] , unweighted_output = True):
        self.cells = []
        self.weights = []
        self.stack_size = stack_size

        for i in range(self.stack_size):
            self.cells.append(lstmCell())
            self.weights.append(random())
        
        if unweighted_output :
            self.weights.pop()
            self.weights[len(weights)-1] = 1
        
        self.tanh = Activation('tanh')
        self.sigmoid = Activation('sigmoid')

    def __call__(self , x):
        for i in range(self.cells):
            x = self.weights[i]*self.cells[i](x)
        return x

    def learn(self , io_pairs = []):
        #[[
        #   Input  : [decimal_1 , decimal_2 ... decimal_n],
        #   Output :  decimal
        # ]]

        """
            -Iterate through input/output pairs
                -Pass inputs through stack
                -Store individual cell output
                -Store individual cell's cell state
                -Store individual cell's hidden state
                
                -calculate loss
                -calculate learning rate
                
                -Iterate through cells from n down to 1
                    - 
        """
        stack_flow = {}
        stack_cell_states = {}
        stack_hidden_states = {}

        for i in io_pairs :
            input_data = i[0]
            output = i[1]

            for j in range(len(input_data)) :
                stack_flow[j] = []
                stack_cell_states[j] = []
                stack_hidden_states[j] = []
                x = input_data[j]
                stack_flow[j].append(x)
                print("Forward propogation :")

                for k in range(len(self.cells)):
                    x = self.cells[k](x)
                    stack_cell_states[j].append(self.cells[k].cell_state)
                    stack_hidden_states[j].append(self.cells[k].hidden_state)
                    stack_flow[j].append(x)
                
                print("     Cell states : " , stack_cell_states[j])
                print("     Hidden states : " , stack_hidden_states[j])

                for k in reversed(range(len(self.cells))):
                    s = self.cells[k].candidate_bias + (self.cells[k].candidate_hidden_weight * stack_hidden_states[j-1][k]) + (self.cells[k].candidate_input_weight * reversed(stack_flow[j])[k+1])
                    t = self.cells[k].input_bias + (self.cells[k].input_hidden_weight * stack_hidden_states[j-1][k]) + (self.cells[k].input_input_weight * reversed(stack_flow[j])[k+1])
                    u = self.cells[k].forget_bias + (self.cells[k].forget_hidden_weight * stack_hidden_states[j-1][k]) + (self.cells[k].forget_input_weight * reversed(stack_flow)[k+1])
                    v = self.cells[k].output_bias + (self.cells[k].output_hidden_weight * stack_hidden_states[j-1][k]) + (self.cells[k].output_input_weight * reversed(stack_flow)[k+1])

                    I_g = self.tanh(s)*self.sigmoid(t)
                    F_g = stack_cell_states[j-1]*self.sigmoid(u)
                    O_g = self.sigmoid(v)
                    o = self.tanh(I_g + F_g)*O_g
                    
                    #remember output of previous cell is input of current cell


#endregionn