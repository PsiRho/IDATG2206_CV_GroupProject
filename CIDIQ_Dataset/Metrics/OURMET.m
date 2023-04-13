function result = OURMET(orig, repr)

py.importlib.import_module('main')

% Convert array
IO = rgb2gray(orig);
IR = rgb2gray(repr);

IO = orig;
IR = repr;

IO_np = py.numpy.array(IO);
IR_np = py.numpy.array(IR);

result = py.main.get_diff(IO_np, IR_np);