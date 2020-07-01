import queue


# Returns a GraphCharacteristics object given a tensorflow graphdef, which has the following properties:
# nodes_by_name: A dictionary that maps node names to node objects in the graph
# node_outputs_by_name: A dictionary that maps node names to their respective output node objects
# input_node_names: List of likely input node names
# input_names: List of likely input nodes
# output_node_names: List of likely output node names
# output_nodes: List of likely output nodes
class GraphCharacteristics:

    def __init__(self, graph_def):
        self.nodes_by_name = {}
        self.node_outputs_by_name = {}

        # Build node-name graph
        for node in graph_def.node:
            self.nodes_by_name[node.name] = node
            self.node_outputs_by_name[node.name] = []

        # Build name-output graph
        for node in graph_def.node:
            for input_node_name in node.input:
                input_node_proper = input_node_name.split(':')[0].lstrip('^')
                self.node_outputs_by_name[input_node_proper].append(node)

        # Ascertain input nodes
        self.input_node_names = [node.name for node in graph_def.node if node.op == 'Placeholder']
        self.input_nodes = [self.nodes_by_name[node_name] for node_name in self.input_node_names]
        print("Input names: ", self.input_node_names)

        # Ascertain output nodes
        self.output_node_names = [node for node in self.node_outputs_by_name.keys() if
                                  self.node_outputs_by_name[node] == []]
        self.output_nodes = [self.nodes_by_name[node_name] for node_name in self.output_node_names]
        print("Output names: ", self.output_node_names)

    # Returns all nodes that are inputs of (but not in) a particular subgraph
    def get_subgraph_inputs(self, subgraph_names):

        if type(subgraph_names) is str:
            subgraph_names = [subgraph_names]

        nodes = []

        for node in self.nodes_by_name.values():
            if not any(x in node.name for x in subgraph_names):
                continue
            for input_node in node.input:
                input_node_proper = input_node.split(':')[0].lstrip('^')
                if not any(x in input_node_proper for x in subgraph_names) and self.nodes_by_name[
                        input_node_proper] not in nodes:
                    nodes.append(self.nodes_by_name[input_node_proper])

        return nodes


# Determines the input dimensions, by whichever means necessary
# TODO: Automatically determine input dimensions for trickier graphs
def get_input_dims(args, input_node):
    proposed_dims = args.input_dims
    node_dims = [input_node.attr['shape'].shape.dim[i].size for i in
                 range(len(input_node.attr['shape'].shape.dim))]

    if proposed_dims is not None:

        if len(proposed_dims) != len(node_dims):
            raise ValueError("Number of provided input dimensions does not match number of existing dimensions. "
                             "Given: {}. Existing: {} "
                             .format(len(proposed_dims), len(node_dims)))

        for idx, dim in enumerate(node_dims):
            if dim != -1 and dim != proposed_dims[idx]:
                raise ValueError(
                    "Provided input dimension {} contradicts existing input dimension. Given: {}. Existing: {}"
                    .format(idx, proposed_dims[idx], dim))

        return proposed_dims

    unknown_dims = node_dims.count(-1)

    if unknown_dims == 0:
        return node_dims

    node_dims_print = [(dim if dim != -1 else '?') for dim in node_dims]
    while True:
        proposed_dims = input("Existing input structure is {}, enter {} missing dimensions:\n"
                              .format(node_dims_print, unknown_dims)).split()
        if len(proposed_dims) != unknown_dims:
            print("Incorrect number of dimensions")
        else:
            break

    proposed_iter = iter(proposed_dims)
    node_dims = [(dim if dim != -1 else int(next(proposed_iter))) for dim in node_dims]
    print("Using input dimensions ", node_dims)
    return node_dims


# Performs a depth-first-search across a tensorflow graph for a node with a name containing the target string
# starting at the node "start"
# note this search from moves outputs to inputs
# target can be either a string or list of strings
# returns a result node if one is found, otherwise returns None
# Will not search more nodes than DFS_SEARCH_LIMIT
def BFS(graph, start, target, graph_chars=None, search_limit=50):
    if graph_chars is None:
        graph_chars = GraphCharacteristics(graph)

    if type(target) == str:
        target = [target]

    target_node = None

    q = queue.Queue()
    searchCount = 0
    q.put(start)
    while q is not None and not q.empty():
        node = q.get()
        for node_name in node.input:
            if any(x in node_name for x in target):
                target_node = graph_chars.nodes_by_name[node_name]
                q = None
                break
            q.put(graph_chars.nodes_by_name[node_name.split(':')[0].lstrip('^')])

        if searchCount > search_limit:
            break

        searchCount += 1

    return target_node


# returns number of classes in SSD classification, where
# graph is a graphDef
# graph_chars is the graph characteristics object, if you've already computed it
def get_num_classes(graph, graph_chars=None):
    import tensorflow as tf

    initial_node_names = ["Postprocessor", "PostProcess"]
    class_node_names = ["ClassPredictor", "TFLite_Detection_PostProcess:1"]

    if graph_chars is None:
        graph_chars = GraphCharacteristics(graph)

    search_nodes = graph_chars.get_subgraph_inputs(initial_node_names)

    class_node = None
    for node in search_nodes:
        if BFS(graph, node, class_node_names, graph_chars=graph_chars) is not None:
            class_node = node
            break
    if class_node is None:
        print("Error: Could not find a class output node for # class determination")
        exit(1)

    # TODO: Remove tensorflow dependency, if possible
    class_tensor = tf.graph_util.import_graph_def(graph, return_elements=[class_node.name + ":0"])

    return int(class_tensor[0].shape[-1])


# Returns the NMS input order, which are the order of [loc_data, conf_data, priorbox_data]
# Where the order is the input order to the postprocessor subgraph with prefix postprocessor_prefix
def get_NMS_input_order(graph, postprocessor_prefix, graph_chars=None):
    if graph_chars is None:
        graph_chars = GraphCharacteristics(graph)

    order = [-1, -1, -1]

    input_nodes = graph_chars.get_subgraph_inputs(postprocessor_prefix)

    # Trim irrelevant inputs
    relevant_input_nodes = []
    for input_node in input_nodes:
        if BFS(graph, input_node, ["BoxEncodingPredictor", "ClassPredictor", "GridAnchor"], graph_chars=graph_chars) \
                is not None and input_node not in relevant_input_nodes:
            relevant_input_nodes.append(input_node)

    if len(relevant_input_nodes) != 3:
        print("NMS input order error: {} relevant input nodes, should be 3".format(len(relevant_input_nodes)))
        return order


    # Find locations
    for idx, node in enumerate(relevant_input_nodes):
        if BFS(graph, node, "BoxEncodingPredictor", graph_chars=graph_chars) is not None:
            order[0] = idx
            break
    if order[0] == -1:
        print("NMS input order error: Could not find the locations input")

    # Find confidences
    for idx, node in enumerate(relevant_input_nodes):
        if BFS(graph, node, "ClassPredictor", graph_chars=graph_chars) is not None:
            order[1] = idx
            break
    if order[1] == -1:
        print("NMS input order error: Could not find the classes input")

    # Find priorboxes
    for idx, node in enumerate(relevant_input_nodes):
        if BFS(graph, node, "GridAnchor", graph_chars=graph_chars) is not None:
            order[2] = idx
            break
    if order[2] == -1:
        print("NMS input order error: Could not find the priorboxes input")

    return order
