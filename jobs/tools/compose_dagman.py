import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose a DAGMan file from a list of arguments")
    parser.add_argument("-n", "--number", type=int, help="Number of jobs to run", required=True)
    parser.add_argument("-f", "--first", type=str, help="First job to run", required=True)
    parser.add_argument("-r", "--rest", type=str, help="Rest of the jobs to run", required=True)
    
    args = parser.parse_args()

    # create list of random names of lenght args.number
    names = [''.join(["job", str(i)]).upper() for i in range(args.number)]

    # write in the dag file
    with open("dagman.dag", "w") as dagman:
        for i, n in enumerate(names):
            dagman.write("JOB {} {}\n".format(n, args.first if i == 0 else args.rest))

        dagman.write("\n")

        for i, n in enumerate(names[:-1]):
            dagman.write("PARENT {} CHILD {}\n".format(n, names[i+1])) 

        for i, n in enumerate(names):
            dagman.write("SCRIPT POST {} post_script.sh\n".format(n))
