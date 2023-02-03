def _read_tsp_data(tsp):
    get_data = open(tsp, 'r')

    name = get_data.readline().strip().split()[1]
    data_type = get_data.readline().strip().split()[1]
    comment = get_data.readline().strip().split()[-1]
    dimension = get_data.readline().strip().split()[1]
    edge_weight_type = get_data.readline().strip().split()[1]
    coordinates = []
    get_data.readline()

    for a in range(int(dimension)):
        x, y = get_data.readline().strip().split()[1:]
        coordinates.append([float(x), float(y)])

    get_data.close()

    return {
        'name': name,
        'type': data_type,
        'comment': comment,
        'dimension': dimension,
        'edge_weight_type': edge_weight_type,
        'coordinates': coordinates
    }

def _display_header_names(dictionary):
    print('\nName: ', dictionary['name'])
    print('Type: ', dictionary['type'])
    print('Comment: ', dictionary['comment'])
    print('Dimension: ', dictionary['dimension'])
    print('Edge Weight Type: ', dictionary['edge_weight_type'], '\n')


def cities_in_array(source_data_path):
    cities = []
    init = _read_tsp_data(source_data_path).get('coordinates')
    for i in range(0, len(init)):
        cities.append(init[i])
    return cities

