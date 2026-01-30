import pyinsim
from MapBuilder import MapGenerator

insim = pyinsim.insim(b'127.0.0.1', 29999, Admin=b'', Flags=pyinsim.ISF_MCI | pyinsim.ISF_LOCAL, Interval=1000)
objects = []
def handle_layout(insim, axm):
    for object in axm.Info:
        index = object.Index
        x = object.X / 16
        y = object.Y / 16
        z = object.Zbyte / 4
        # Coordinates in Meters:
        objects.append((index, x, y, z))
    print(len(objects))
    if len(objects) >= 155:
        gen = MapGenerator(objects)
        gen.process()
        gen.save_to_json()
        gen.debug_plot()

insim.bind(pyinsim.ISP_AXM, handle_layout)
insim.send(pyinsim.ISP_TINY, ReqI=255, SubT=pyinsim.TINY_AXM)

pyinsim.run()