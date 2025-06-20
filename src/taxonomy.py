class TaxonomyNode:
    def __init__(self, name: str, description: str = "", depth: int = 0):
        self.name = name
        self.description = description
        self.children: dict[str, TaxonomyNode] = {}
        self.depth = depth

    def add_child(self, child: 'TaxonomyNode') -> None:
        """
        Add a child node under this node.
        """
        self.children[child.name] = child

    def get_child(self, name: str) -> 'TaxonomyNode | None':
        """
        Retrieve a direct child by name.
        """
        return self.children.get(name)

    def to_dict(self) -> dict:
        """
        Convert this node and its subtree into a nested dictionary.
        """
        return {
            'name': self.name,
            'description': self.description,
            'depth': self.depth,
            'children': [child.to_dict() for child in self.children.values()]
        }

    def find(self, name: str) -> 'TaxonomyNode | None':
        """
        Recursively search for a node by name in this subtree.
        """
        if self.name == name:
            return self
        for child in self.children.values():
            found = child.find(name)
            if found:
                return found
        return None

    def find_path(self, name: str, path: list[str] | None = None) -> list[str] | None:
        """
        Recursively find the path from this node to the node with the given name.
        Returns list of node names or None if not found.
        """
        if path is None:
            path = []
        path.append(self.name)
        if self.name == name:
            return path.copy()
        for child in self.children.values():
            result = child.find_path(name, path)
            if result:
                return result
        path.pop()
        return None

    def print_tree(self, indent: int = 0) -> None:
        """
        Print the subtree in a human-readable tree format.
        """
        print(' ' * indent + f"- {self.name} (Depth: {self.depth}): {self.description}")
        for child in self.children.values():
            child.print_tree(indent + 4)


class Taxonomy:
    def __init__(self):
        self.root = TaxonomyNode('root', 'Root of taxonomy', depth=0)

    @property
    def depth(self) -> int:
        """
        Calculate the maximum depth of the taxonomy.
        """
        def get_max_depth(node: TaxonomyNode) -> int:
            if not node.children:
                return node.depth
            return max(get_max_depth(child) for child in node.children.values())
        return get_max_depth(self.root)

    def add_node(self, path: list[str], description: str = "") -> None:
        """
        Add a node to the taxonomy under the given path.
        `path` is a list of names leading from root to the new node (including new node name at end).
        """
        current = self.root
        for i, name in enumerate(path):
            child = current.get_child(name)
            if not child:
                # The depth of the new node is the depth of its parent + 1
                child = TaxonomyNode(name, depth=current.depth + 1)
                current.add_child(child)
            current = child
        current.description = description or current.description

    def get_node(self, path: list[str]) -> TaxonomyNode | None:
        """
        Retrieve a node by its full path list of names.
        """
        current = self.root
        for name in path:
            current = current.get_child(name)
            if not current:
                return None
        return current

    def get_children(self, path: list[str] | None = None) -> list[TaxonomyNode]:
        """
        List direct children of the node at the given path (or root if path is None).
        """
        node = self.root if path is None else self.get_node(path)
        return list(node.children.values()) if node else []

    def find(self, name: str) -> TaxonomyNode | None:
        """
        Global search for a node by name.
        """
        return self.root.find(name)

    def find_path(self, name: str) -> list[str] | None:
        """
        Find the hierarchical path (names) from root to the named node.
        """
        return self.root.find_path(name)

    def to_dict(self) -> dict:
        """
        Convert the entire taxonomy to a nested dictionary.
        """
        return self.root.to_dict()

    def load_from_dict(self, data: dict) -> None:
        """
        Load taxonomy structure from a nested dictionary as produced by `to_dict`.
        """
        def _build(node_data: dict) -> TaxonomyNode:
            node = TaxonomyNode(
                node_data['name'],
                node_data.get('description', ''),
                node_data.get('depth', 0)
            )
            for child_data in node_data.get('children', []):
                child_node = _build(child_data)
                node.add_child(child_node)
            return node

        self.root = _build(data)

    def print_tree(self) -> None:
        """
        Print the full taxonomy tree.
        """
        self.root.print_tree()

if __name__ == '__main__':
    # Example usage:
    taxonomy = Taxonomy()
    taxonomy.add_node(['Science'], 'All scientific papers')
    taxonomy.add_node(['Science', 'AI'], 'Artificial Intelligence')
    taxonomy.add_node(['Science', 'AI', 'NLP'], 'Natural Language Processing')
    taxonomy.add_node(['Science', 'Biology'], 'The study of living organisms')
    taxonomy.add_node(['Humanities', 'History'], 'The study of past events')


    print("--- Taxonomy Tree with Depth ---")
    taxonomy.print_tree()

    print("\n--- Taxonomy Properties ---")
    # Get the overall depth of the taxonomy
    print(f"The total depth of the taxonomy is: {taxonomy.depth}")

    # Find a specific node and get its depth
    nlp_node = taxonomy.find('NLP')
    if nlp_node:
        print(f"The depth of the 'NLP' node is: {nlp_node.depth}")

    history_node = taxonomy.get_node(['Humanities', 'History'])
    if history_node:
        print(f"The depth of the 'History' node is: {history_node.depth}")