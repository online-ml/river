def line(text):
    return f"{text}\n"


def h1(text):
    return line(f"# {text}")


def h2(text):
    return line(f"## {text}")


def h3(text):
    return line(f"### {text}")


def h4(text):
    return line(f"#### {text}")


def link(caption, href):
    return f"[{caption}]({href})"


def code(text):
    return f"`{text}`"


def li(text):
    return f"- {text}\n"
