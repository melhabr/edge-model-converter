# Builds two dictionaries and two lists,
# one which links names to nodes,
# one which links names to names of output nodes
# one which has the list of input nodes, and
# one which has the list of output ndoes
def analyze_graph(graph_def):
    names = {}
    outputs = {}

    # Build node-name graph
    for node in graph_def.node:
        names[node.name] = node
        outputs[node.name] = []

    # Build name-output graph
    for node in graph_def.node:
        for input_node_name in node.input:
            input_node_proper = input_node_name.split(':')[0].lstrip('^')
            outputs[input_node_proper].append(node)

    # Ascertain input nodes
    input_nodes = [node.name for node in graph_def.node if node.op == 'Placeholder']
    print("Input names: ", input_nodes)

    # Ascertain output nodes
    output_nodes = [node for node in outputs.keys() if outputs[node] == []]
    print("Output names: ", output_nodes)

    return names, outputs, input_nodes, output_nodes

# Determines the input dimensions, by whichever means neccessary
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
