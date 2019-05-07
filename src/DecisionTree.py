from src.InfoGain import best_info_gain
from src.Colors import Colors as c


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

    def __init__(self, dataset):
        super(DecisionTree, self).__init__()
        self.dataset = dataset
        self.Tree = None
        self.error = None
        self.create()

    def create(self):
        """
            Public method to create a tree from the loaded dataset
        """
        try:
            training_data = self.dataset.data
            attributes = self.dataset.attributes.copy()
            self.Tree = self.__pvt_build_tree_recursive(
                training_data, attributes)
        except Exception as e:
            self.error = e

    def __pvt_build_tree_recursive(self, D, L):
        """
            Internal method that implements the recursive creation algorithm
        """
        # Steps are listed according to the
        # given algorithm especification (pt-br)
        # https://moodle.inf.ufrgs.br/pluginfile.php/135382/mod_resource/content/1/ArvoresDeDecisao.pdf
        # Slide 40

        # 1. Cria Nó N
        new_node = _Node()

        # 2. Se todos os exemplos em D possuem a mesma classe y,
        # então retorna N como um nó folha rotulado com y.
        class_to_compare = D[0][self.dataset.predictclass]
        all_classes_equal = True
        for i in range(len(D)):
            if (class_to_compare != D[i][self.dataset.predictclass]):
                all_classes_equal = False
        if all_classes_equal is True:
            new_node.set_label(class_to_compare, True)
            return new_node

        # 3. Se L é vazia, então retorn N como um nó folha
        # rotulado com a classe y mais frequente em D.

        if len(L) == 0:
            new_node.set_label(get_most_commom(
                D, self.dataset.predictclass), True)
            return new_node

        # 4.Senão
        # 4.1 A = atributo que apresenta melhor critério de divisão.
        # Esse cálculo será realizado utilizando o método de ganho
        # de informação.
        A = best_info_gain(D, L, self.dataset.predictclass)
        # 4.2 Associe A ao nó N
        new_node.set_label(A)
        # 4.3 L = L - {A}
        L.remove(A)

        # 4.4 Para cada valor v distinto do atributo A,
        #     considerando os exemplos em D.

        # counter armazena um dict com os atributos únicos
        counter = {}
        for i in range(len(D)):
            counter[D[i][A]] = True

        for key, val in counter.items():
            # 4.4.1 Dv = subconjunto dos dados de treinamento em que A = v'
            Dv = [x for x in D if (x[A] == key)]
            # 4.4.2 Se Dv vazio, então retorn N como um nó folha rotulado com
            #       a classe yi mais frequente em Dv.
            if len(Dv) == 0:
                new_node.set_label(get_most_commom(
                    Dv, self.dataset.predictclass))
                return new_node
            # 4.4.3 Senão, associe N a uma subárvore retornada por f(Dv,L)
            else:
                new_node.child[key] = self.__pvt_build_tree_recursive(Dv, L)
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
        if(node.labeled):
            retval += "   %s:%s\n" %(self.dataset.predictclass, c.OKGREEN+node.label+c.ENDC)
        else:
            retval += 'label: %s\n' % (c.OKBLUE+node.label+c.ENDC)
        if(len(node.child) > 0):
            for i in range(space * self.__div):
                retval += ' '
            retval += 'options:\n'
            space += 1
            for child in node.child:
                for i in range(space * self.__div):
                    retval += ' '
                retval += '%s'%(c.HEADER+child+c.ENDC)

                retval += self.__print(node.child[child], space)
        else:
            pass

        return retval


class _Node(object):
    """Node of a tree Class"""

    def __init__(self):
        super(_Node, self).__init__()
        self.labeled = bool()
        self.label = str()
        self.child = dict()

    def set_label(self, label, labeled=False):
        self.label = label
        self.labeled = labeled
