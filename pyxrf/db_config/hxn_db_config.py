try:
    from databroker.v0 import Broker
except ModuleNotFoundError:
    from databroker import Broker

db = Broker.named("hxn")
from hxntools.handlers import register
register(db)

#from hxntools.handlers.xspress3 import Xspress3HDF5Handler
#from hxntools.handlers.timepix import TimepixHDF5Handler
#
#db = Broker.named("hxn")
# db_analysis = Broker.named('hxn_analysis')

#db.reg.register_handler(Xspress3HDF5Handler.HANDLER_NAME, Xspress3HDF5Handler, overwrite=True)
#db.reg.register_handler(TimepixHDF5Handler._handler_name, TimepixHDF5Handler, overwrite=True)
