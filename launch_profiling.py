#!venv/bin/python3
import os
from os import listdir
from os.path import isfile, join
import readline
import blake3
from tinydb import TinyDB, Query


PATH_CAIRO_PROGRAMS  = "tests/cairo_programs"
CAIRO_FILES = [f for f in listdir(PATH_CAIRO_PROGRAMS) if isfile(join(PATH_CAIRO_PROGRAMS, f)) if f[-6:]=='.cairo']
LIB_CAIRO_FILES = [f for f in listdir("lib") if isfile(join("lib", f)) if f[-6:]=='.cairo']

def get_hash_if_file_exists(file_path:str)-> str:
    isExist = os.path.exists(file_path)
    if isExist==False:
        return None
    else:
        json_bytes = open(file_path, "rb")
        bytes = json_bytes.read()
        hash = blake3.blake3(bytes).digest()
        return str(hash)

def mkdir_if_not_exists(path:str):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print(f"Directory created : {path} ")

mkdir_if_not_exists("tests/profiling")
mkdir_if_not_exists("build")

def complete(text,state):
    volcab = CAIRO_FILES
    results = [x for x in volcab if x.startswith(text)] + [None]
    return results[state]

readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

FILENAME_DOT_CAIRO = input('\n>>> Enter cairo program to run or double press for suggestions <TAB> : \n\n')
FILENAME = FILENAME_DOT_CAIRO.removesuffix('.cairo')

mkdir_if_not_exists(f"tests/profiling/{FILENAME}")

def write_all_hash(db:TinyDB):
    for FILE_DOT_CAIRO in CAIRO_FILES:
        db.insert({'name':FILE_DOT_CAIRO, 'hash':get_hash_if_file_exists(f"{PATH_CAIRO_PROGRAMS}/{FILE_DOT_CAIRO}")})
    for FILE_DOT_CAIRO in LIB_CAIRO_FILES:
        db.insert({'name':FILE_DOT_CAIRO, 'hash':get_hash_if_file_exists(f"lib/{FILE_DOT_CAIRO}")})

def get_all_hash():
    r = []
    for FILE_DOT_CAIRO in CAIRO_FILES:
        r.append(get_hash_if_file_exists(f"{PATH_CAIRO_PROGRAMS}/{FILE_DOT_CAIRO}"))
    for FILE_DOT_CAIRO in LIB_CAIRO_FILES:
        r.append(get_hash_if_file_exists(f"lib/{FILE_DOT_CAIRO}"))
    return r 


db = TinyDB(f"{PATH_CAIRO_PROGRAMS}/programs_hash.json")

if len(db)!=(len(CAIRO_FILES)+len(LIB_CAIRO_FILES)):
    db.remove(Query().name!=0)
    write_all_hash()


hash_table = db.all()
current_hash_table = get_all_hash()


def did_some_file_changed():
    for f,h in zip(hash_table,current_hash_table):
        # print(f, h )
        if f["hash"]!=h:
            return True
    return False

prev_hash = get_hash_if_file_exists(f"build/{FILENAME}.json")

if did_some_file_changed() or os.path.exists(f"build/{FILENAME}.json")==False:
    print(f"Compiling {FILENAME_DOT_CAIRO} because some files have changed ... ")

    os.system(f"cairo-compile {PATH_CAIRO_PROGRAMS}/{FILENAME_DOT_CAIRO} --output build/{FILENAME}.json")
else:
    print(f"Skipping compilation for {FILENAME_DOT_CAIRO} since no file has changed.")


new_hash = get_hash_if_file_exists(f"build/{FILENAME}.json")


if new_hash!=prev_hash:
    print(f"Running {FILENAME_DOT_CAIRO} ...")

    os.system(f"cairo-run --program=build/{FILENAME}.json --layout=all --profile_output ./tests/profiling/{FILENAME}/profile.pb.gz")
    print(f"Running profiling tool for {FILENAME_DOT_CAIRO} because the compiled file has changed ... ")

    os.system(f"cd ./tests/profiling/{FILENAME} && go tool pprof -png profile.pb.gz ")
else:
    print(f"Running {FILENAME_DOT_CAIRO} without profiling ...")

    os.system(f"cairo-run --program=build/{FILENAME}.json --layout=all")

    print(f"Profiling for {FILENAME_DOT_CAIRO} should already be available in /tests/profiling/{FILENAME} ! ")




