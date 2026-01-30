import time

import pyinsim

class LfsConnectionTest:
    def __init__(self):
        self.connection_successful = False
        self.insim = pyinsim.insim(b'127.0.0.1', 29999, Admin=b'')

    def insim_state(self, insim, sta):
        self.connection_successful = True
        pyinsim.closeall()

    def run_test(self):
        # Init new InSim object.
        # Bind ISP_STA packet to insim state method.
        self.insim.bind(pyinsim.ISP_STA, self.insim_state)
        self.insim.send(pyinsim.ISP_TINY, ReqI=255, SubT=pyinsim.TINY_SST)
        # Start pyinsim.
        pyinsim.run()
        return self.connection_successful



if __name__ == '__main__':
    test = LfsConnectionTest()
    test.run_test()
