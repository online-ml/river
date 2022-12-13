import json

import dominate
from dominate.tags import *
from slugify import slugify
from watermark import watermark

from river import datasets

with open("../docs/benchmarks/index.md", "w", encoding='utf-8') as f:
    print_ = lambda x: print(x, file=f, end="\n\n")
    print_(
        """
        ---
        hide:
        - navigation
        ---
        """
    )
    print_("# Benchmarks")

    print_(
