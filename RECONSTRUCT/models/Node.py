'''
 AMR Node Object
'''

class Node(object):
    def __init__(self, name='', ful_name='', next_nodes=[], parents = set(), edge_label='',
                 is_entity=False, entity_type='', entity_name='', wiki='',
                 polarity=False, content='',original_content = ''):
        self.name = name               # Node name (acronym)
        self.ful_name = ful_name       # Full name of the node
        self.next_nodes = next_nodes   # Next nodes (list)
        self.parents = parents
        self.edge_label = edge_label   # Edge label between two nodes
        self.is_entity = is_entity     # Whether the node is named entity
        self.entity_type = entity_type # Entity type
        self.entity_name = entity_name # Entity name
        self.wiki = wiki               # Entity Wikipedia title
        self.polarity = polarity       # Whether the node is polarity
        self.content = content         # Original content
        self.original_content = original_content

    def __str__(self):
        if not self.ful_name:
            name = 'NODE NAME: %s\n' % self.name
        else:
            name = 'NODE NAME: %s / %s\n' % (self.name, self.ful_name)
        polarity = 'POLARITY: %s\n' % self.polarity
        children = 'LINK TO:\n'
        parents = 'LINKED BY:\n'
        for parent in self.parents:
            parents += '\t%s --(%s)-> %s\n' %(parent.name, parent.edge_label, self.name)

        for i in self.next_nodes:
            if not i.ful_name:
                children += '\t(%s) -> %s\n' % (i.edge_label, i.name)
            else:
                children += '\t(%s) -> %s / %s\n' % \
                            (i.edge_label, i.name, i.ful_name)
        if not self.is_entity:
            return name + polarity + children + parents
        else:
            s = 'ENTITY TYPE: %s\nENTITY NAME: %s\nWIKIPEDIA TITLE: %s\n' % \
                (self.entity_type, self.entity_name, self.wiki)
            return name + polarity + s + children + parents
