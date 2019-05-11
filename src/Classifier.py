def Classifier(entry, tree):
    return __recursive_classifier(entry, tree.Tree, tree.numerical_attributes)


def __recursive_classifier(entry, node, numerical_attributes):
    node_attribute = node.label

    if node_attribute not in numerical_attributes:
        # Categorical Attribute
        # If the node does not know the value for this attribute of the
        # entry, then its value is called 'except'
        try:
            value_for_this_attribute = entry[node_attribute]
            node.child[value_for_this_attribute]
        except KeyError:
            value_for_this_attribute = 'except'
        if node.is_leaf:
            return node.label
        else:
            return __recursive_classifier(entry,
                                          node.child[value_for_this_attribute],
                                          numerical_attributes)
    else:
        # Numerical Attribute
        print('TODO: Calculate category for numerical attribute')
        return None
