import xml.etree.ElementTree as ET


class AspectCClass:
    """
    C++ class information provided by the AspectC++ repo.acp

    :attr name: class name (str)
    :attr class_id: internal AspectC++ class ID (str/int)
    :attr functions: functions implemented by this class (list of :class:`AspectCFunction`)
    :attr function: dict mapping function name to :class:`AspectCFunction`. Only sensible for classes which do not overload functions.
    """

    def __init__(self, name, class_id, functions):
        self.name = name
        self.class_id = class_id
        self.functions = functions
        self.function = dict()

        for function in self.functions:
            self.function[function.name] = function


class AspectCFunction:
    """
    C++ function informationed provided by the AspectC++ repo.acp

    :attr name: function name (str)
    :attr kind: function kind (str/int)
    :attr function_id: internal AspectC++ function ID (str/int)
    :attr argument_types: C++ types of function arguments (list of str)
    :attr return_type: C++ type of function return value (str)
    """

    def __init__(self, name, kind, function_id, argument_types, return_type):
        """
        Create C++ function description.

        :param name: function name (str)
        :param kind: function kind (str/int)
        :param function_id: internal AspectC++ function ID (str/int)
        :param argument_types: C++ types of function arguments (list of str)
        :param return_type: C++ type of function return value (str)
        """
        self.name = name
        self.kind = kind
        self.function_id = function_id
        self.argument_types = argument_types
        self.return_type = return_type

    @classmethod
    def from_function_node(cls, function_node):
        """
        Create C++ function description from AspectC++ repo.acp Function node

        :param function_node: `xml.etree.ElementTree.Element` node
        """
        name = function_node.get("name")
        kind = function_node.get("kind")
        function_id = function_node.get("id")
        return_type = None
        argument_types = list()
        for type_node in function_node.findall("result_type/Type"):
            return_type = type_node.get("signature")
        for type_node in function_node.findall("arg_types/Type"):
            argument_types.append(type_node.get("signature"))
        return cls(
            name=name,
            kind=kind,
            function_id=function_id,
            argument_types=argument_types,
            return_type=return_type,
        )


class Repo:
    """
    C++ class and function information provided by the AspectC++ repo.acp

    :attr classes: list of C++ classes
    :attr classes_by_name: dict of C++ classes by name
    """

    def __init__(self, repo_path):
        """
        Load repo.acp.

        :param repo_path: path to repo.acp
        """
        self.repo_path = repo_path
        self.tree = ET.parse(repo_path)
        self.root = self.tree.getroot()
        self._load_classes()

    def _load_classes(self):
        self.classes = list()
        for class_node in self.root.findall(
            'root/Namespace[@name="::"]/children/Class'
        ):
            name = class_node.get("name")
            class_id = class_node.get("id")
            functions = list()
            for function_node in class_node.findall("children/Function"):
                function = AspectCFunction.from_function_node(function_node)
                functions.append(function)
            self.classes.append(AspectCClass(name, class_id, functions))

        self.class_by_name = dict()
        for class_data in self.classes:
            self.class_by_name[class_data.name] = class_data
