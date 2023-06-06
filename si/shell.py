
import os
import sys
import pwd

try:
    wd = os.getcwd()
except OSError:
    wd = os.path.expanduser("~")

try:
    import customtkinter as ctk
    from packaging import version as vr

    req_version = '5.1.3'
    if vr.parse(ctk.__version__) < vr.parse(req_version):
        print("Please upgrade CustomTkinter to version 5.1.3 or higher...")
        sys.exit(1)

    app = ctk.CTk()
    
    if not app.winfo_exists():
        print("Display not available...")
        sys.exit(1)

    app.mainloop()

except ModuleNotFoundError:
    print("Please install CustomTkinter...")
    sys.exit(1)

from si_lib.version import APP_FULLNAME,APP_NAME,APP_VERSION





