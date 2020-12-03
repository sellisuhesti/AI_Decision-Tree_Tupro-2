import numpy as np
import math

class Node(object):
    #buat node decision tree pake struktur data
    def __init__(self, atribut=None, value=None, pointer=False, level=False):
        self.atribut = atribut
        self.value = value
        self.left = None
        self.right = None
        self.pointer = pointer
        self.level = level

class DecisionTree(object):
    #buat desicion tree
    def __init__(self, root):
        self.root = root
        self.kedalaman = self.kedalaman_tree()

    def kedalaman_tree(self):
        count = 0
        if self.root is not None and not self.root.terminal_node:
            count = self.jumlahNodes(self.root)
        return count

    def jumlahNodes(self, node): #hitung jumlah node
        count = 1
        if node.left is not None and not node.left.pointer:
            count += self.jumlahNodes(node.left)
        if node.right is not None and not node.right.pointer:
            count += self.jumlahNodes(node.right)
        return count


class evtree(object):

    def __init__(self, p_crossover=0.6, p_mutation=0.4, p_split=0.7, p_prune=0.1, populasi = 400, max_kedalaman=1023, i = 500):
        self.p = [] 
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_split = p_split
        self.p_prune = p_prune
        self.populasu = populasi
        self.i = i
        self.max_kedalaman = float(max_kedalaman)

        self.num_attributes = None

        self.best_candidates = []

    def fit(self, x, y):
        self.num_attributes = x.shape[1]

       


    def test(self, X):
        pass

    def treeDepth(self, tree):
        #hitung kedalaman tree
        pass

    def split_points(self, values):
        #hitung split tree
        return (values[1:] + values[:-1])/2.0

    def get_threshold(self, attr_idx, X, y):
        #batas untuk split antara root dan node
        vals = np.sort(X[:, attr_idx])
        thresholds = self.split_points(vals)

        threshold = np.random.choice(thresholds, 1)[0]
        return threshold

    def most_common_label(self, labels):
        # buat label terbaru dan  nge return label baru
        unique_elements = list(set(labels))
        max_freq = 0
        common_element = unique_elements[0]
        for element in unique_elements:
            if (labels.count(element) > max_freq):
                max_freq = labels.count(element)
                common_element = element
        return common_element

    def insert_children(self, node, X, y, attrs, attr_idx, split_value, terminal=False):
        # buat node sekaligus ngasih value child
        attrs = attrs[:]
        if not terminal:
            X_left = X[X[:, attr_idx] <= split_value]
            y_left = y[X[:, attr_idx] <= split_value]

            X_right = X[X[:, attr_idx] > split_value]
            y_right = y[X[:, attr_idx] > split_value]

            # node sebelah kiri dari root
            split_attribute_left = attrs.pop(np.random.randint(0, len(attrs)))
            attr_idx_left = split_attribute_left.keys()[0]
            split_value_left = self.get_threshold(attr_idx_left, X_left, y_left)
            node.left = Node(split_attribute_left, split_value_left)

            # node sebelah kanan dari root
            split_attribute_right = attrs.pop(np.random.randint(0, len(attrs)))
            attr_idx_right = split_attribute_right.keys()[0]
            split_value_right = self.get_threshold(attr_idx_right, X_right, y_right)
            node.right = Node(split_attribute_right, split_value_right)

            # get data cabang
            node.left = self.insert_children(node.left, X_left, y_left, attrs, attr_idx_left, split_value_left, True)
            node.right = self.insert_children(node.right, X_right, y_right, attrs, attr_idx_right, split_value_right, True)
            return node

        else:
            random_label = np.random.choice(y, 1)[0]
            y_left = y[X[:, attr_idx] <= split_value]
            node.left = Node(None, random_label, terminal_node=True)

            random_label = np.random.choice(y, 1)[0]
            y_right = y[X[:, attr_idx] > split_value]
            node.right = Node(None, random_label, terminal_node=True)
            return node

    def create_initial_tree(self, X, y, attrs):
        X = X[:]
        y = y[:]
        attrs = attrs[:]
        split_attribute = attrs.pop(np.random.randint(0, len(attrs)))
        attr_idx = split_attribute.keys()[0]
        split_value = self.get_threshold(attr_idx, X, y)
        root = Node(split_attribute, split_value)
        root = self.insert_children(root, X, y, attrs, attr_idx, split_value, False)

        # split node sebelah kiri dengan kemungkinan / probabilitas p_split
        if np.random.random() < self.p_split:
            root.left = self.insert_children(root.left, X, y, attrs, attr_idx, split_value, False)

        # split node sebelah kanan dengan kemungkinan / probabilitas p_split
        if np.random.random() < self.p_split:
            root.right = self.insert_children(root.right, X, y, attrs, attr_idx, split_value, False)

        #self.print_tree(root)
        return root

    def print_tree(self, root):
        # print tree
        node = root
        print (node.attribute, node.value)
        node = root.left
        print (node.attribute, node.value)
        node = root.right
        print (node.attribute, node.value)
        node = root.left.left
        print (node.attribute, node.value)
        node = root.left.right
        print (node.attribute, node.value)

    def proto(self, X, y, attrs):
        root1 = self.create_initial_tree(X, y, attrs)
        tree1 = DecisionTree(root1)
        print (tree1.depth)

        new_tree = self.mutate(tree1, tree1.depth-1, X, y, attrs)

        print (new_tree.depth)

    def initialization(self, X, y, attrs):
        X = X[:]
        y = y[:]
        attrs = attrs[:]
        # inisialisasi populasi dari tree
        for _ in xrange(self.population_size):
            root1 = self.create_initial_tree(X, y, attrs)
            tree1 = DecisionTree(root1)
            accuracy = self.evaluate(tree1, X, y)
            depth = tree1.depth
            fitness_score = self.fitness(accuracy, depth)
            self.population.append((tree1, fitness_score))

    def fitness(self, accuracy, depth):
        #hitung fitness
        alpha1 = 0.99 
        alpha2 = 0.01 

        depth_score = depth / self.target_depth
        fitness_score = (alpha1*accuracy) + (alpha2*(1-depth_score))
        return fitness_score

    def evaluate(self, tree, X, y):
        # evaluasi akurasi fitness dengan child 
        X = X[:]
        y = y[:]
        m, n = X.shape
        correct = 0

        for idx in xrange(m):
            input_feature = X[idx, :]
            dependent = y[idx]

            node = tree.root
            while node is not None:
                if node.terminal_node:
                    if node.value == dependent:
                        correct += 1
                    break
                attr_idx = node.attribute.keys()[0]
                feature_val = input_feature[attr_idx]
                if feature_val <= node.value:
                    node = node.left
                else:
                    node = node.right

        return correct/float(m)

    def mutate(self, tree, node_idx, X, y, attrs):
        #mutasi
        X = X[:]
        y = y[:]
        attrs = attrs[:]
        split_attribute = attrs.pop(np.random.randint(0, len(attrs)))
        attr_idx = split_attribute.keys()[0]
        split_value = self.get_threshold(attr_idx, X, y)

        head = tree.root
        if node_idx == 0:
            head.attribute = split_attribute
            head.value = split_value
            return DecisionTree(head)

        count_idx = 0
        ptr = head

        queue1 = []
        queue1.append(ptr)

        while len(queue1) > 0:
            ptr = queue1.pop(0)
            if not ptr.terminal_node:
                count_idx += 1
                if count_idx == node_idx:
                    break
                if ptr.left is not None and not ptr.left.terminal_node:
                    queue1.append(ptr.left)
                if ptr.right is not None and not ptr.right.terminal_node:
                    queue1.append(ptr.right)

        ptr.attribute = split_attribute
        ptr.value = split_value
        return DecisionTree(head)

    def crossover(self, tree1, node_idx1, tree2, node_idx2):
        if node_idx1 == 0:
            return tree2

        count_idx1 = 1 #pengganti
        count_idx2 = 0 #pointer parent yang bakal di ganti

        head1 = tree1.root
        head2 = tree2.root

        ptr1 = head1
        ptr2 = head2

        queue1 = []
        queue1.append(ptr1)

        while len(queue1) > 0:
            ptr1 = queue1.pop(0)
            if not ptr1.terminal_node:
                count_idx1 += 1
                if count_idx1 >= node_idx1:
                    break
                if ptr1.left is not None and not ptr1.left.terminal_node:
                    queue1.append(ptr1.left)
                if ptr1.right is not None and not ptr1.right.terminal_node:
                    queue1.append(ptr1.right)

        if node_idx2 == 0:
            if ptr1.left is not None and not ptr1.left.terminal_node:
                ptr1.left = ptr2
            else:
                ptr1.right = ptr2
            return DecisionTree(head1)

        queue2 = []
        queue2.append(ptr2)

        while len(queue2) > 0:
            ptr2 = queue2.pop(0)
            if not ptr1.terminal_node:
                count_idx2 += 1
                if count_idx2 == node_idx2:
                    break
                if ptr2.left is not None and not ptr2.left.terminal_node:
                    queue2.append(ptr1.left)
                if ptr2.right is not None and not ptr2.right.terminal_node:
                    queue2.append(ptr1.right)

        if ptr1.left is not None and not ptr1.left.terminal_node:
            ptr1.left = ptr2
        else:
            ptr1.right = ptr2

        return DecisionTree(head1)


    def survivor_selction(self, X, y, attrs):
        X = X[:]
        y = y[:]
        attrs = attrs[:]
        sorted_population = self.population[:]
        sorted_population.sort(key=itemgetter(1), reverse=True)
        next_generation = sorted_population[:3]  
        fitness_scores = np.array([_[1] for _ in self.population])
        fitness_scores = fitness_scores / float(sum(fitness_scores))

        population_idxs = xrange(self.population_size)
        parents = np.random.choice(population_idxs, 5000, True, fitness_scores)
        idx = 0
        while len(next_generation) < self.population_size:
            parent1 = self.population[parents[idx]][0]
            idx += 1
            node_idx1 = np.random.randint(0, parent1.depth-1)
            if np.random.random() < self.p_mutation:
                child = self.mutate(parent1, node_idx1, X, y, attrs)
            else:
                parent2 = self.population[parents[idx]][0]
                idx += 1
                node_idx2 = np.random.randint(0, parent2.depth-1)
                child = self.crossover(parent1, node_idx1, parent2, node_idx2)

            accuracy = self.evaluate(child, X, y)
            depth = child.depth
            fitness_score = self.fitness(accuracy, depth)
            next_generation.append((child, fitness_score))
        self.population = next_generation



if __name__ == '__main__':
   
    data = np.genfromtxt('data.txt', delimiter=',')
    X = data[1:, :-1]
    y = data[1:, -1]

    attributes = [{idx: attr.strip()} for idx, attr in enumerate(header.split(","))]
    #print attributes
    outcome = attributes[-1].values()[0]
    attributes = attributes[:-1]
    et = evtree()
    #print et.get_threshold(2, X, y)
    et.initialization(X, y, attributes)
    et.survivor_selction(X, y, attributes)
 
