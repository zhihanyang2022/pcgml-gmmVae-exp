import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path_to_nb", help="path to the to-be-converted notebook")
parser.add_argument("path_to_py", help="path to the exported python script")
args = parser.parse_args()

def not_empty(cell): return len(cell['source']) != 0

def is_code(cell): return cell['cell_type'] == 'code'

def to_be_exported(cell): return cell['source'][0].startswith('#export')

def nb2py(nb_fname:str, py_fname:str):
    
    with open(nb_fname, 'r') as json_f:
        nb = json.load(json_f)
        
    import_cells = []
    generic_cells = []
    classes = {}
    static_methods2classes = {}
    for cell in nb['cells']:
        if not_empty(cell) and is_code(cell):
            if to_be_exported(cell):
                
                tags_string = cell['source'][0]
                cell_without_export_tag = cell['source'][1:]
                
                tags = tags_string.split(' ')
                tags.pop(0)
                if len(tags) > 0:  # with arguments
                    for tag in tags:
                        tag = tag[1:]  # remove the preceding "-"
                        tag_type, arg_value = tag.split(':')
                        if arg_value[-1] == '\n': arg_value = arg_value[:-1]
                        if tag_type == 'class':
                            
                            if arg_value not in classes.keys(): 
                                classes[arg_value] = [cell_without_export_tag]
                            else:
                                classes[arg_value].append(cell_without_export_tag)
                            
                            method_name = cell_without_export_tag[0].split('(')[0][4:]
                            static_methods2classes[method_name] = arg_value 
                                
                else:  # without arguments
                    if cell['source'][1:][0].split(' ')[0] in ('import', 'from'):
                        import_cells.append(cell_without_export_tag)
                    else:
                        generic_cells.append(cell_without_export_tag) 
    
    with open(py_fname, 'w') as py_f:
        
        for cell in import_cells:
            for line in cell:
                py_f.write(line)
            py_f.write('\n\n')
        
        for class_name, static_methods in classes.items():
            py_f.write(f"class {class_name}():\n\n")
            for method in static_methods:
                py_f.write('    @staticmethod\n')
                for line in method:
                    for m, c in static_methods2classes.items():
                        if m + '.(' in line:
                            index = line.index(m)
                            line = line[:index] + c + '.' + line[index:]
                    py_f.write(f'    {line}')
                py_f.write('\n\n')
        
        for cell in generic_cells:
            for line in cell:
                py_f.write(line)  # write a single line of code
            py_f.write('\n\n')  # last line of a cell does not have the newline character, so two is needed here

nb2py(args.path_to_nb, args.path_to_py)

