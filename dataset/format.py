file = open("/home/ka/SecGCN/dataset/cora/cora.content", 'r')
output = open("/home/ka/SecGCN/dataset/cora/features.txt", 'w')
output_labels = open("/home/ka/SecGCN/dataset/cora/labels.txt", 'w')
internal_node_id = {}
label_id = {}
id_counter = 0
labels_id_counter = 0
for line in file.readlines():
    parts = line.strip().split()
    node_id = int(parts[0])
    features = list(map(int, parts[1:-1]))
    label = parts[-1]
    if node_id not in internal_node_id:
        internal_node_id[node_id] = id_counter
        id_counter += 1
    if label not in label_id:
        label_id[label] = labels_id_counter
        labels_id_counter += 1
    output.write(' '.join(map(str, features)) + '\n')
    output_labels.write(f"{label_id[label]}\n")
file.close()
output.close()
output_labels.close()

file = open("/home/ka/SecGCN/dataset/cora/cora.cites", 'r')
output = open("/home/ka/SecGCN/dataset/cora/edges.txt", 'w')
for line in file.readlines():
    parts = line.strip().split()
    src_id = int(parts[0])
    dst_id = int(parts[1])
    if src_id in internal_node_id and dst_id in internal_node_id:
        output.write(f"{internal_node_id[src_id]} {internal_node_id[dst_id]}\n")
file.close()
output.close()