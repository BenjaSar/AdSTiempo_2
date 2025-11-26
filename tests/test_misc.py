from utils.misc import print_box

print_box("This is a test message to demonstrate the print_box function.")
print_box() # should print an empty line
long_text = "This is a very long message that should be split into multiple lines because it exceeds the maximum line length specified in the function."
for i in range(50,60):
    print_box(long_text,i,margin=2)