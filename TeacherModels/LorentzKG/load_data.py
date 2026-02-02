import numpy as np
import os
import sys

def save_dict_to_file(dictionary, filename):
    """Save a dictionary to a file, one key-value pair per line."""
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}\t{value}\n")


class Data:
    def __init__(self, data_dir="data/WN18RR/"):
        self.data_filter = None
        self.train_data = self.load_data(data_dir, "train", self.data_filter)
        self.valid_data = self.load_data(data_dir, "valid", self.data_filter)
        self.test_data = self.load_data(data_dir, "test", self.data_filter)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.get_relations(self.data)
        self.relation2id = {}
        self.entity2id = {}
        self.init_relation2id(data_dir=data_dir)
        self.init_entity2id(data_dir=data_dir)
        # for i, rel in enumerate(self.relations):
        #     self.relation2id[rel] = i
        # for i, ent in enumerate(self.entities):
        #     self.entity2id[ent] = i
        self.pos_entity = [set() for _ in range(len(self.relations))]
        for triplet in self.data:
            self.pos_entity[self.relation2id[triplet[1]]].add(self.entity2id[triplet[2]])
        for i in range(len(self.pos_entity)):
            self.pos_entity[i] = list(self.pos_entity[i])
        
        # save_dict_to_file(self.relation2id, 'relation2id.dict')
        # save_dict_to_file(self.entity2id, 'entity2id.dict')
        # sys.exit(0)


    def load_data(self, data_dir, data_type="train", data_filter=None):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]

        if data_filter is not None:
            data = [i for i in data if i[1] == data_filter]

        data_rev = [[i[2], i[1] + "_reverse", i[0]] for i in data] # 在这里你就加入负关系了啊
        data += data_rev
        print("data is: ", len(data))
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    def init_relation2id(self, data_dir):
        relation_file = os.path.join(data_dir, "relations.dict")
        with open(relation_file, 'r') as file:
            for line in file:
                value, key = line.strip().split('\t')
                self.relation2id[key] = int(value)
        
        relation_num = len(self.relation2id)
        with open(relation_file, 'r') as file:
            for line in file:
                value, key = line.strip().split('\t')
                self.relation2id[key + "_reverse"] = int(value) + relation_num
        
    
    def init_entity2id(self, data_dir):
        entity_file = os.path.join(data_dir, "entities.dict")
        with open(entity_file, 'r') as file:
            for line in file:
                value, key = line.strip().split('\t')
                self.entity2id[key] = int(value)