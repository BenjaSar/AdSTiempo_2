"""
Utility functions for miscellaneous tasks
"""
import re

def _split_lines(text, max_length):
    """
    Split a long text into multiple lines not exceeding max_length
    """
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:
            if current_line:
                current_line += " "
            current_line += word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line: lines.append(current_line)

    return lines

def _print_line(text, INNER_SPACE, margin=2):
    """
    Format a single line to fit inside the box and print it
    """
    padding = INNER_SPACE - len(text) - margin*2
    left_padding = padding // 2 + margin
    right_padding = padding - padding // 2 + margin
    print("║" + " " * (left_padding) + text + " " * right_padding + "║")

def check_breaks(text):
    """
    Check for leading and trailing newlines in the text:
        1. Leading newlines: one or more '\n' at the start
        2. Trailing newlines: one or more '\n' at the end
        3. Empty string check and handle accordingly
    """
    lines = text.split("\n")
    # Check for empty lines
    verification = [True if line != "" else False for line in lines]
    empty_string = True if all(not check for check in verification) and len(lines) > 1 else False
    
    # Detect leading and trailing newlines
    top_spacing, bottom_spacing = 0, 0
    detected_top = re.search(r'^\n+', text) # matches newlines at the start
    detected_bottom = re.search(r'\n+$', text) # matches newlines at the end
    if detected_bottom: bottom_spacing += len(detected_bottom.group())
    if detected_top: top_spacing += len(detected_top.group())
    # Remove leading and trailing newlines from lines
    cleaned_lines = lines[top_spacing:(None if bottom_spacing == 0 else -bottom_spacing)]

    # Adjust spacing if the entire string is empty/newlines only
    if empty_string:
        bottom_spacing = bottom_spacing // 2
        top_spacing = top_spacing - bottom_spacing
        cleaned_lines = []
    
    return cleaned_lines, top_spacing, bottom_spacing 

def print_box(text="", LINE_LENGTH=81, margin=2, vertical_padding=0):
    """
    Print a message inside a decorative box. If the message is too long,
    it will be split into multiple lines. When no message is provided,
    a horizontal line is printed instead.

    Args:
        text (str): The message to print inside the box (multi-line supported).
        LINE_LENGTH (int): Total length of the box including borders.
        margin (int): Number of spaces to leave as margin on each side of the text.
        vertical_padding (int): Number of empty lines to add above and below the text.  
    """
    INNER_SPACE = LINE_LENGTH - 2
    HEADER = "╔" + "═"*INNER_SPACE + "╗"
    V_PADDING = "║" + " " * INNER_SPACE + "║"
    FOOTER = "╚" + "═"*INNER_SPACE + "╝"

    lines, t_breaks, b_breaks = check_breaks(text)

    if text == "" or text == None:
        print("═" + "═"*INNER_SPACE + "═")
    else:
        print("\n" * t_breaks, end="")  # Leading newlines
        print(HEADER) # Top border
        [print(V_PADDING) for _ in range(vertical_padding)] # Vertical spacing

        # Block of text (multi-line supported)
        for line_text in lines:
            line_length = len(line_text) + margin*2
            # split into multiple lines if necessary
            if line_length > INNER_SPACE:
                chunks = _split_lines(line_text, INNER_SPACE-margin*2)
                for chunk in chunks:
                    _print_line(chunk, INNER_SPACE, margin)
            else:
                _print_line(line_text, INNER_SPACE, margin)

        [print(V_PADDING) for _ in range(vertical_padding)] # Vertical spacing
        print(FOOTER) # Bottom border
        print("\n" * b_breaks, end="")  # Leading newlines
