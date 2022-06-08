from typing import Union
from xml.etree import ElementTree as ET

from river.tree.base import Branch, Leaf


def tree_to_html(tree: Branch) -> ET.Element:
    def add_node(node: Union[Branch, Leaf], parent: ET.Element):

        # We're building this:
        #
        # <li>
        #   <pre>Split information</pre>
        #   <ul>...</ul
        # <li>

        li = ET.Element("li")
        parent.append(li)  # type: ignore

        code = ET.Element("code")

        if isinstance(node, Branch):
            code.text = node.repr_split
            li.append(code)

            ul = ET.Element("ul")

            for child in node.children:
                add_node(node=child, parent=ul)

            li.append(ul)

        else:
            code.text = repr(node)
            li.append(code)

    root = ET.Element("ul", attrib={"class": "tree"})
    add_node(node=tree, parent=root)

    return root


CSS = """
.tree,
.tree ul,
.tree li {
    list-style: none;
    margin: 0;
    padding: 0;
    position: relative;
}

.tree {
    margin: 0 0 1em;
    text-align: center;
}

.tree,
.tree ul {
    display: table;
}

.tree ul {
    width: 100%;
}

.tree li {
    display: table-cell;
    padding: .5em 0;
    vertical-align: top;
}

.tree li:before {
    outline: solid 1px #666;
    content: "";
    left: 0;
    position: absolute;
    right: 0;
    top: 0;
}

.tree li:first-child:before {
    left: 50%;
}

.tree li:last-child:before {
    right: 50%;
}

.tree code,
.tree span {
    border: solid .1em #666;
    display: inline-block;
    margin: 0 .2em .5em;
    padding: .2em .5em;
    position: relative;
}

.tree ul:before,
.tree code:before,
.tree span:before {
    outline: solid 1px #666;
    content: "";
    height: .5em;
    left: 50%;
    position: absolute;
}

.tree ul:before {
    top: -.5em;
}

.tree code:before,
.tree span:before {
    top: -.55em;
}

.tree>li {
    margin-top: 0;
}

.tree>li:before,
.tree>li:after,
.tree>li>code:before,
.tree>li>span:before {
    outline: none;
}
"""
