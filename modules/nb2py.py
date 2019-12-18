import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path_to_nb", help="path to the to-be-converted notebook")
parser.add_argument("path_to_py", help="path to the exported python script")
args = parser.parse_args()

def not_empty(cell): return len(cell['source']) != 0

def is_code(cell): return cell['cell_type'] == 'code'

def to_be_exported(cell): return cell['source'][0].startswith('#export')

def load_json_from_nb(nb_path):
    with open(nb_path, 'r') as nb_f:
        json_dict = json.load(nb_f)
    return json_dict

def nb2py(nb_path:str, py_path:str):
    
    nb = load_json_from_nb(nb_path)
        
    module_level_comments = []
    import_cells = []
    generic_cells = []
    classes = {}  # map classes to a list of static methods that fall under them
    static_methods2classes = {}  # map static methods to the classes under which they fall
    
    for cell in nb['cells']:  # loop over cells
        if not_empty(cell) and is_code(cell):  # only process the non-empty code cells
            
            if to_be_exported(cell):  # do not move this line to its parent if statement
                
                tags_string = cell['source'][0]
                cell_without_export_tag = cell['source'][1:]
                
                tags = tags_string.split(' ')
                tags.pop(0)  # remove the preceding "#export"
                if len(tags) > 0:  # if tags actually contain arguments
                    
                    for tag in tags:  # loop over tags
                        
                        tag = tag[1:]  # remove the preceding "-"
                        tag_type, arg_value = tag.split(':')  # parse a argument-value pair
                        if arg_value[-1] == '\n': arg_value = arg_value[:-1]  # preprocess the last argument-value pair
                        
                        if tag_type == 'class':
                            
                            class_name = arg_value
                            
                            if class_name not in classes.keys(): 
                                classes[class_name] = [cell_without_export_tag]
                            else:
                                classes[class_name].append(cell_without_export_tag)
                            
                            method_name = cell_without_export_tag[0].split('(')[0][4:]
                            static_methods2classes[method_name] = class_name 
                                
                else:  # if tags do not contain arguments
                    
                    if cell['source'][1:][0] == '"""\n':
                        module_level_comments.append(cell_without_export_tag)
                    elif cell['source'][1:][0].split(' ')[0] in ('import', 'from'):
                        import_cells.append(cell_without_export_tag)
                    else:
                        generic_cells.append(cell_without_export_tag) 
    
    with open(py_path, 'w') as py_f:
        
        for cell in module_level_comments:
            for line in cell:
                py_f.write(line)
            py_f.write('\n\n')
        
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
                        if ('def ' + m not in line) and ((m + '(' in line) or ('=' + m in line)):
                            index = line.index(m)
                            line = line[:index] + c + '.' + line[index:]
                    py_f.write(f'    {line}')
                py_f.write('\n\n')
        
        for cell in generic_cells:
            for line in cell:
                py_f.write(line)  # write a single line of code
            py_f.write('\n\n')  # last line of a cell does not have the newline character, so two is needed here

nb2py(args.path_to_nb, args.path_to_py)

