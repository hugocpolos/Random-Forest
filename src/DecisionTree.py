from src.InfoGain import best_info_gain
from src.Colors import Colors as c
from math import sqrt
from random import sample


def get_most_commom(D, target_class):
    """
        Return the most common attribute of the target_class
        in a dataset D.
    """
    counter = {}
    for i in range(len(D)):
        counter[D[i][target_class]] = 0
    for i in range(len(D)):
        counter[D[i][target_class]] += 1
    max_val = 0
    most_commom = ""
    for key, value in counter.items():
        if value > max_val:
            most_commom = key
            max_val = value
    return most_commom


class DecisionTree(object):
    """
    Decision Tree Class.
    Stores a Decision Tree generated from a Dataset Object.
    Args:
        dataset (Dataset Object): Dataset object used as Training Data

    Attributes:
        dataset (Dataset Object): List with the dataset attributes
        Tree (internal _Node class): None if the tree wasn't created yet,
                                     stores the Tree otherwise.
        error (exception class): is None if the construction or the
                                execution of the last method was successful,
                                otherwise stores the Exception that crashed it.
    """

    def __init__(self, dataset, attributes, target_attribute,
                 numerical_attributes, single_tree_print=False):
        super(DecisionTree, self).__init__()
        self.dataset = dataset
        self.attributes = attributes.copy()
        self.target_attribute = target_attribute
        self.numerical_attributes = numerical_attributes
        self.m = int(sqrt(len(self.attributes)))
        self.single_tree_print_mode = single_tree_print
        self.iteration_count = 0
        self.Tree = None
        self.error = None
        self.__calculate_numerical_cut_value__()
        self.create()

    def create(self):
        """
            Public method to create a tree from the loaded dataset
        """
        try:
            self.Tree = self.__pvt_build_tree_recursive(self.dataset.copy())
        except Exception as e:
            print(e)
            self.error = e

    def __calculate_numerical_cut_value__(self):
        for attrib in self.attributes:
            if attrib in self.numerical_attributes:
                avg_value = 0
                for value in self.dataset:
                    avg_value += float(value[attrib])
                avg_value /= (len(self.dataset))
                self.numerical_attributes[attrib] = avg_value

    def __attribute_sampling(self):
        self.m = int(sqrt(len(self.attributes)))
        return [self.attributes[i]
                for i in sample(range(0, len(self.attributes)), self.m)]

    def __pvt_build_tree_recursive(self, D):
        """
            Internal method that implements the recursive creation algorithm
        """
        # Steps are listed according to the
        # given algorithm especification (pt-br)
        # https://moodle.inf.ufrgs.br/pluginfile.php/135382/mod_resource/content/1/ArvoresDeDecisao.pdf
        # Slide 40
        if(self.single_tree_print_mode):
            print('----------------------------------------------------------')
            print("iteration %d:" % (self.iteration_count))
            self.iteration_count += 1
            if(len(self.attributes) > 0):
                print("Available Attributes:\n%s" % (self.attributes))
            else:
                print('Leaf node')

        L = self.__attribute_sampling()
        self.__calculate_numerical_cut_value__()

        if(self.single_tree_print_mode and len(self.attributes) > 0):
            print('Attribute sampling:\n%s' % (L))

        # 1. Cria Nó N
        new_node = _Node()

        # 2. Se todos os exemplos em D possuem a mesma classe y,
        # então retorna N como um nó folha rotulado com y.
        class_to_compare = D[0][self.target_attribute]
        all_classes_equal = True
        for i in range(len(D)):
            if (class_to_compare != D[i][self.target_attribute]):
                all_classes_equal = False
        if all_classes_equal is True:
            new_node.set_label(class_to_compare, is_leaf=True)
            if(self.single_tree_print_mode):
                print('label: %s' % (new_node.label))
            return new_node

        # 3. Se L é vazia, então retorn N como um nó folha
        # rotulado com a classe y mais frequente em D.

        if len(L) == 0:
            new_node.set_label(get_most_commom(
                D, self.target_attribute), is_leaf=True)
            if(self.single_tree_print_mode):
                print('label: %s' % (new_node.label))
            return new_node

        # 4.Senão
        # 4.1 A = atributo que apresenta melhor critério de divisão.
        # Esse cálculo será realizado utilizando o método de ganho
        # de informação.
        if(self.single_tree_print_mode):
            print('Cálculo de Ganho de informação:')
        A = best_info_gain(D, L, self.target_attribute,
                           self.numerical_attributes, single_tree_print=self.single_tree_print_mode)

        if(self.single_tree_print_mode):
            print('\'%s\' Selected' % (A))
        # 4.2 Associe A ao nó N
        new_node.set_label(A)
        # 4.3 L = L - {A}
        self.attributes.remove(A)

        # 4.4 Para cada valor v distinto do atributo A,
        #     considerando os exemplos em D.
        counter = {}

        # Trecho para tratar atributos numericos #
        if (A in self.numerical_attributes):
            # 4.4.1 Lesser = subconjunto dos dados de treinamento em que
            # A é menor que o valor de corte
            Lesser = [x for x in D if (
                float(x[A]) < self.numerical_attributes[A])]
            # 4.4.1 Greater = subconjunto dos dados de treinamento em que
            # A é maior que o valor de corte
            Greater = [x for x in D if (
                float(x[A]) >= self.numerical_attributes[A])]

            if (len(Greater) is 0 or len(Lesser) is 0):
                new_node.set_label(get_most_commom(
                    D, self.target_attribute), is_leaf=True)
                return new_node

            # 4.4.2 Se Lesser ou Greater for vazio,
            # então retorn N como um nó folha rotulado com
            # a classe yi mais frequente em D.

            new_node.child['<' + str(self.numerical_attributes[A])] = self.__pvt_build_tree_recursive(
                Lesser)

            new_node.child['>=' + str(self.numerical_attributes[A])] = self.__pvt_build_tree_recursive(
                Greater)
            # exit(0)
        else:  # counter armazena um dict com os atributos únicos
            for i in range(len(D)):
                counter[D[i][A]] = True

            for key, val in counter.items():
                # 4.4.1 Dv = subconjunto dos dados de treinamento em que A = v'
                Dv = [x for x in D if (x[A] == key)]
                # 4.4.2 Se Dv vazio, então retorn N como um nó folha
                # rotulado com a classe yi mais frequente em D.
                if len(Dv) == 0:
                    new_node.set_label(get_most_commom(
                        D, self.target_attribute), is_leaf=True)
                    return new_node

                # 4.4.3 Senão, associe N a uma subárvore retornada por f(Dv,L)
                else:
                    new_node.child[key] = self.__pvt_build_tree_recursive(
                        Dv)
                    # Heurística, cria um nó chamado except,
                    # que atende a qualquer valor do atributo
                    # que o treinamento não conhece.
                    new_node.child['except'] = _Node()
                    new_node.child['except'].set_label(get_most_commom(
                        D, self.target_attribute), is_leaf=True)
        # 4.5 Retorne N
        return new_node

    #
    #
    # Print Methods
    #
    def __str__(self):
        self.__div = 10
        return "\nTREE:\n\n" + self.__print(self.Tree)

    def __print(self, node, space=0):
        retval = '\n'
        for i in range(space * self.__div):
            retval += ' '
        if node is None:
            print('fim')
            exit(0)
        if(node.is_leaf):
            retval += "   %s:%s\n" % (self.target_attribute,
                                      c.OKGREEN + node.label + c.ENDC)
        else:
            retval += 'label: %s\n' % (c.OKBLUE + node.label + c.ENDC)
        if(len(node.child) > 0):
            for i in range(space * self.__div):
                retval += ' '
            retval += 'options:\n'
            space += 1
            for child in node.child:
                for i in range(space * self.__div):
                    retval += ' '
                retval += '%s' % (c.HEADER + child + c.ENDC)

                retval += self.__print(node.child[child], space)
        else:
            pass

        return retval


class _Node(object):
    """Node of a tree Class
        is_leaf: if the node is a leaf node or no
    """

    def __init__(self):
        super(_Node, self).__init__()
        self.is_leaf = bool()
        self.label = str()
        self.child = dict()

    def set_label(self, label, is_leaf=False):
        self.label = label
        self.is_leaf = is_leaf
