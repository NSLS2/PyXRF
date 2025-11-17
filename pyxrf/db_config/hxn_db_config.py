try:
    from databroker.v0 import Broker
except ModuleNotFoundError:
    from databroker import Broker

from hxntools.handlers.timepix import TimepixHDF5Handler
from hxntools.handlers.xspress3 import Xspress3HDF5Handler

# FIXME: this broker needs to be updated to DataBroker v2,
# and ensure all proper data format handlers are registered and used,
# such as is done from hxntools here.
# One possible option:
#
#     import databroker
#     db = databroker.catalog['hxn']
#
# But we have to make sure the server has correctly configured the
# data loading handlers.

db = Broker.named("hxn")
# db_analysis = Broker.named('hxn_analysis')

db.reg.register_handler(Xspress3HDF5Handler.HANDLER_NAME, Xspress3HDF5Handler, overwrite=True)
db.reg.register_handler(TimepixHDF5Handler._handler_name, TimepixHDF5Handler, overwrite=True)
