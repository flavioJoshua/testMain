import os
import sys
import logging
import gettext
from optparse import OptionParser
from gettext import gettext as _

import gtk

sys.path.insert(0, '/usr/share/disk-manager')

from DiskManager.DiskManager import *
from DiskManager.Config import Config

def main(args, opts) :
    if opts.version :
    print PACKAGE,VERSION
    return

if opts.query :
if not os.getuid() == 0 :
logging.warning("Query database without root privilege.")
logging.warning("Result might be incomplete.\n")
info = get_diskinfo_backend()()
print info.export(opts.query.strip())
return
gtk.window_set_default_icon_name("disk-manager")

if not os.getuid() == 0 :
dialog("warning", _("Insufficient rights"), \
_("You need administrative rights to start this application."))
return

app = DiskManager()
app.run()

def get_opt_args() :

parser = OptionParser()
parser.add_option ("-v", "--version", help="show version of the application and exit.", \
action="store_true", dest="version", default=False)
parser.add_option ("-q", "--query-database", type="string", 
dest="query", default="", metavar="DEVICE",
help="query database for DEVICE.\nSet DEVICE to all to print full database.")
parser.add_option ("-d", "--debug", action="store_true",
dest="debug", default=False,
help="print debugging information.")

return parser.parse_args()

if __name__ == '__main__' :

gettext.bindtextdomain("disk-manager",localedir)
gettext.textdomain("disk-manager")
(opts, args) = get_opt_args()
if opts.debug :
    level = logging.DEBUG
else :
    level = logging.INFO
logging.basicConfig(level=level, format='%(levelname)s : %(message)s')
main(args, opts)
logging.shutdown()
-------