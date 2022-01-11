import subprocess
import os
import argparse
import time

FROM = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"
TO = "/eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/big_data/"

parser = argparse.ArgumentParser(description='Move files to EOS')
parser.add_argument('--filename', type=str, help='Filename to move', required=True)

args = parser.parse_args()

# check if the file is in the directory
print("Looking for file: " + args.filename)

if os.path.isfile(os.path.join(FROM, args.filename)):
    print("File found, moving to EOS")

    start = time.time()
    # move the file to EOS with eos cp
    try:
        subprocess.call(["eos", "cp", os.path.join(FROM, args.filename), TO])
    except Exception as e:
        print("Error moving file: " + str(e))
    end = time.time()
    # print time in hh:mm:ss
    print("Time taken: " + str(time.strftime("%H:%M:%S", time.gmtime(end - start))))
    # check if the file is in the EOS directory
    print("Looking for file in EOS: " + args.filename)

    if os.path.isfile(os.path.join(TO, args.filename)):
        print("File found, deleting from local directory")

        # delete the file from the local directory
        os.remove(os.path.join(FROM, args.filename))

        print("File deleted, Terminating...")
    else:
        print("File not found in EOS, Terminating...")

else:
    print("File not found! Terminating...")
