import os
import socket
from nipype.interfaces.matlab import MatlabCommand

if socket.gethostname() == 'malin':
    os.environ['MATLABCMD'] = "/opt/matlab/R2015b/bin/matlab -nodesktop -nosplash"
    MatlabCommand.set_default_paths('/opt/matlab/R2015b/toolbox/spm12')
    MatlabCommand.set_default_matlab_cmd("/opt/matlab/R2015b/bin/matlab -nodesktop -nosplash")
    TPM = '/opt/matlab/R2015b/toolbox/spm12/tpm/TPM.nii'
    # os.environ['MATLABCMD'] = "/opt/matlab/R2012a/bin/matlab -nodesktop -nosplash"
    # MatlabCommand.set_default_paths('/opt/matlab/R2012a/toolbox/spm12')
    # MatlabCommand.set_default_matlab_cmd("/opt/matlab/R2012a/bin/matlab -nodesktop -nosplash")
elif socket.gethostname() == 'cala':
    os.environ['MATLABCMD'] = "/opt/matlab/64bit/R2015a/bin/matlab -nodesktop -nosplash"
    MatlabCommand.set_default_paths('/opt/matlab/64bit/R2015a/toolbox/spm12')
    MatlabCommand.set_default_matlab_cmd("/opt/matlab/64bit/R2015a/bin/matlab -nodesktop -nosplash")
    TPM = '/opt/matlab/64bit/R2015a/toolbox/spm12/tpm/TPM.nii'

def display_crash_files(crashfile, rerun=False):
    from nipype.utils.filemanip import loadcrash
    crash_data = loadcrash(crashfile)
    node = crash_data['node']
    tb = crash_data['traceback']
    print("\n")
    print("File: %s"%crashfile)
    print("Node: %s"%node)
    if node.base_dir:
        print("Working directory: %s" % node.output_dir())
    else:
        print("Node crashed before execution")
    print("\n")
    print("Node inputs:")
    print(node.inputs)
    print("\n")
    print("Traceback: ")
    print(''.join(tb))
    print("\n")

    if rerun:
        print("Rerunning node")
        node.base_dir = None
        node.config['crashdump_dir'] = '/tmp'
        node.run()
        print("\n")